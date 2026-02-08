from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math
import numpy as np

from os.path import join as pjoin

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from loguru import logger

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage

from data.maps_utils import logits_to_onehot
from metrics.pta_difference import PTADifferenceMetric
from metrics.dice import MultiClassDiceCollection, compute_dice

from .helper import RobustIdentity
from .model_ensemble import PlCommonModuleEnsemble

import torch_optimizer as topt
import torch.optim as toptorch


logger = logging.getLogger(__name__)

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    #def __init__(self, config, vis):
    def __init__(self, config_transformer, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config_transformer["num_heads"]
        self.attention_head_size = int(config_transformer['hidden_size'] / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config_transformer['hidden_size'], self.all_head_size)
        self.key = Linear(config_transformer['hidden_size'], self.all_head_size)
        self.value = Linear(config_transformer['hidden_size'], self.all_head_size)

        self.out = Linear(config_transformer['hidden_size'], config_transformer['hidden_size'])
        self.attn_dropout = Dropout(config_transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config_transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights



class Mlp(nn.Module):
    def __init__(self, config_transformer):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config_transformer['hidden_size'], config_transformer["mlp_dim"])
        self.fc2 = Linear(config_transformer["mlp_dim"], config_transformer['hidden_size'])
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config_transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self,channels_in, channels_out, strides=[[1,1], [1,1]], normalization=nn.BatchNorm2d, normalization_params={}, activation=nn.ReLU, activation_params={"inplace": True}):
        super().__init__()
        #Si usamos normalizacion entonces el bloque de convolucion es
        if "num_features" not in normalization_params:
            normalization_params = normalization_params.copy()
            normalization_params["num_features"] = channels_out
        self.model = nn.Sequential(
            #primer sub-bloque
            nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=strides[0], padding=1),
            normalization(**normalization_params),
            activation(**activation_params),
                
            #segundo sub-bloque
            nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=strides[1], padding=1),
            normalization(**normalization_params),
            activation(**activation_params)
        )
        
    
    def forward(self, x):
        return self.model(x)        


class FeatureExtractor(nn.Module):
    # Inspired in https://amaarora.github.io/2020/09/13/unet.html
    def __init__(self, channels = (1, 64, 128, 256, 512, 1024), pool = False, pool_params={"kernel_size": 2, 'stride': 2}, conv_config={}):
        super().__init__()
        conv_list = [ConvBlock(channels[0], channels[1], **conv_config)]
        if pool:
            self.pool = nn.MaxPool2d(**pool_params)
            conv_list.extend([ConvBlock(channels[i+1], channels[i+2], **conv_config) for i in range(len(channels)-2)])
        else:
            self.pool = None
            # We replace pooling with a [2,2] stride in the first convolution of each block except for the first one
            conv_list.extend([ConvBlock(channels[i+1], channels[i+2], strides=[[2,2], [1,1]], **conv_config) for i in range(len(channels)-2)])

        self.down_blocks = nn.ModuleList(conv_list)

    def forward(self, x):
        outputs = []
        for i, b in enumerate(self.down_blocks):
            x = b(x)
            outputs.append(x)
            if self.pool is not None:
                x = self.pool(x)
        return outputs


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.config = config
        img_size = _pair(img_size)

        config_transformer = config['transformer']

        config_unet = config['unet']
        pool = config_unet.get("pool", False)
        channels = config_unet['channels']
        

        if config_transformer['patches'].get("grid") is not None:
            grid_size = config_transformer['patches']["grid"]
            patch_size = (1, 1) #_pair(config_transformer['patches']["size"])
            patch_size_real = (patch_size[0] * 32, patch_size[1] * 32)
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])  
            self.hybrid = True
        else:
            patch_size = _pair(config_transformer['patches']["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = FeatureExtractor(channels=channels, pool=pool, pool_params={"kernel_size": 2, 'stride': 2}, conv_config=config_unet["conv_config"])

        self.patch_embeddings = Conv2d(in_channels=channels[-1],
                                       out_channels=config_transformer['hidden_size'],
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config_transformer['hidden_size']))

        self.dropout = Dropout(config_transformer["dropout_rate"])


    def forward(self, x):
        if self.hybrid:
            enc_output = self.hybrid_model(x)
        x = self.patch_embeddings(enc_output[-1])  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)

        
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, enc_output


class Block(nn.Module):
    '''
	Clase que representa a un bloque del Transformer Encoder: MSA + MLP 
	'''
    def __init__(self, config_transformer, vis):
        super(Block, self).__init__()
        self.hidden_size = config_transformer['hidden_size']
        self.attention_norm = LayerNorm(config_transformer['hidden_size'], eps=1e-6)
        self.ffn_norm = LayerNorm(config_transformer['hidden_size'], eps=1e-6)
        self.ffn = Mlp(config_transformer)
        self.attn = Attention(config_transformer, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, mlp_dim, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()


            self.ffn.fc1.weight.copy_(mlp_weight_0[:mlp_dim, :])
            self.ffn.fc2.weight.copy_(mlp_weight_1[:, :mlp_dim])
            self.ffn.fc1.bias.copy_(mlp_bias_0[:mlp_dim])
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class ViT(nn.Module):
    def __init__(self, config_transformer, vis):
        super(ViT, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config_transformer['hidden_size'], eps=1e-6)
        for _ in range(config_transformer["num_layers"]):
            layer = Block(config_transformer, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Encoder(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Encoder, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.vit = ViT(config['transformer'], vis)

    def forward(self, input_ids):
        embedding_output, enc_outputs = self.embeddings(input_ids)
        encoded, attn_weights = self.vit(embedding_output)  # (B, n_patch, hidden)
        return encoded, attn_weights, enc_outputs


class Conv2dIntermedia(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dIntermedia, self).__init__(conv, bn, relu)



class DecoderUNet(nn.Module):
    def __init__(self, config, channels=(1024, 512, 256, 128, 64), conv_config={}):
        super().__init__()
        self.conv_intermedia = Conv2dIntermedia(
            config['transformer']['hidden_size'],
            channels[1],
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        conv_list = [nn.UpsamplingBilinear2d(scale_factor=2) for i in range(len(channels)-1)] #esto no reduce el numero de mapas, solo aumenta las dimensiones de los mapas al doble
        self.trans_conv = nn.ModuleList(conv_list)
        self.up_blocks = nn.ModuleList([ConvBlock(2*channels[i+1], channels[i+1]//2, **conv_config) for i in range(len(channels)-1)])

    def forward(self, x, enc_outputs):
        B, n_patch, hidden = x.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_intermedia(x)

        for i in range(len(self.up_blocks)):
            x = self.trans_conv[i](x)
            x = torch.cat([x, enc_outputs[i]], dim=1)
            x = self.up_blocks[i](x)
        return x


def get_class_from_string(name):
    modules = {
        "torch_optimizer": topt,
        "torch.optim": toptorch
    }
    module_name, class_name = name.rsplit(".", 1)
    return getattr(modules[module_name], class_name)


class TransUNet(pl.LightningModule):
    def __init__(self, config, loss_config, optim_config, zero_head=False, vis=False):
        super().__init__()
        self.config_unet = config['unet']
        self.config_transformer = config['transformer']

        classes = self.config_unet['classes']
        num_classes = len(classes)
        self.num_classes = num_classes

        self.img_size = self.config_transformer['img_size']

        self.zero_head = zero_head
        self.classifier = self.config_transformer["classifier"]

        channels = self.config_unet["channels"]

        self.encoder = Encoder(config, self.img_size, vis)
        self.decoder = DecoderUNet(config, channels=channels[1:][::-1], conv_config=self.config_unet["conv_config"])
        self.out_conv = nn.Conv2d(16, num_classes, kernel_size=3, padding=1)

        self.non_linearity = config.get("non_linearity", RobustIdentity())

        self.loss = loss_config["function"](**loss_config["params"])
        self.loss_channel_dim = loss_config.get("channel_dim", False)
        self.loss_non_linearity = loss_config.get("non_linearity", RobustIdentity())

        self.optim_class = optim_config["function"]
        self.optim_params = optim_config["params"]

        self.dice = MultiClassDiceCollection(classes, prefix="val/")
        self.pta_difference = PTADifferenceMetric(classes)


    def forward(self, x):
        encoded, attn_weights, enc_outputs = self.encoder(x)
        x = self.decoder(encoded, enc_outputs[:-1][::-1])
        x = self.out_conv(x)
        x = self.non_linearity(x, dim=1)
        return x

    def compute_loss(self, output, target):
        if self.loss_channel_dim:
            # We add a new dimension if required by the loss. It does not affect Dice or PTA computation
            target = torch.unsqueeze(target, 1)

        loss = self.loss(self.loss_non_linearity(output, dim=1), target)

        return loss

    def training_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["mask"]
        output = self.forward(x)
    
        loss = self.compute_loss(output, y)
    
        # Registra bajo dos nombres:
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)         # <- para que aparezca en barra
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=True) # <- para los logs y tensorboard
    
        return loss



    def validation_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["mask"]
        output = self.forward(x)

        loss= self.compute_loss(output, y)

        logits = F.softmax(output, dim=1)
        self.dice(*logits_to_onehot(logits, y, self.num_classes))
        self.pta_difference(logits, y)

        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict(self.dice, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val/pta_difference',self.pta_difference, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["mask"]
        output = self.forward(x)

        return {
            "original_idx": batch["idx"],
            "output": output,
            "logits": F.softmax(output, dim=1),
            "target": y
        }
        


    def configure_optimizers(self):
        optim_class = self.optim_class
        optim_params = self.optim_params.copy()

        if isinstance(optim_class, str):
            optim_class = get_class_from_string(optim_class)

        if optim_class.__name__ == "Lookahead":
            base_class = optim_params.pop("optimizer")
            if isinstance(base_class, str):
                base_class = get_class_from_string(base_class)

            base_params = {k: v for k, v in optim_params.items() if k in ["lr", "weight_decay"]}
            lookahead_params = {k: v for k, v in optim_params.items() if k in ["k", "alpha"]}

            base_optim = base_class(self.parameters(), **base_params)
            return optim_class(base_optim, **lookahead_params)

        else:
            return optim_class(self.parameters(), **optim_params)

    #pasar como parametro un valor booleano only_emb que indique
    #si solo cargamos los embeddings y posiciones
    def load_from(self, weights):
        with torch.no_grad():

            res_weight = weights
            self.encoder.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.encoder.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

            self.encoder.vit.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.encoder.vit.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

            posemb_new = self.encoder.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.encoder.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1]-1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.encoder.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                if self.classifier == "seg":
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                #print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.encoder.embeddings.position_embeddings.copy_(np2th(posemb))

            # Encoder whole
            for bname, block in self.encoder.vit.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, self.config_transformer['mlp_dim'], n_block=uname)


class TransUNet2DEnsemble(TransUNet):
    def __init__(self, checkpoints, **kwargs):
        pl.LightningModule.__init__(self)

        device = kwargs["device"]
        self.model = PlCommonModuleEnsemble(TransUNet, kwargs, checkpoints, device=device)

        net_config = kwargs["config"]
        optim_config = kwargs["optim_config"]
        loss_config = kwargs["loss_config"]

        classes = net_config['unet']["classes"]
        self.num_classes = len(classes)
        self.non_linearity = net_config.get("non_linearity", RobustIdentity())

        self.loss = loss_config["function"](**loss_config["params"])
        self.loss_channel_dim = loss_config.get("channel_dim", False)
        self.loss_non_linearity = loss_config.get("non_linearity", RobustIdentity())

        self.optim_class = optim_config["function"]
        self.optim_params = optim_config["params"]

        self.dice = MultiClassDiceCollection(classes, prefix="val/")
        self.pta_difference = PTADifferenceMetric(classes)

    def forward(self, x):
        logger.info(f"[DEBUG] Forward called. Input device: {x.device}, Model device: {next(self.model.parameters()).device}")
        x = self.model(x)
        x = self.non_linearity(x, dim=1)
        return x
