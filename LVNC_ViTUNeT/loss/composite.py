import torch
import torch.nn as nn
from loss.lovasz import LovaszSoftmaxLoss
from loss.dice_focal import DiceFocalLoss

class CombinedLoss(nn.Module):
    def __init__(
        self,
        lovasz_params=None,
        focal_params=None,
        lovasz_weight=0.5,
        focal_weight=0.5,
        normalize_weights=True,
        log_partial_losses=False
    ):
        super().__init__()
        
        # Si no se pasan parametros, usamos valores por defecto
        lovasz_params = lovasz_params or {"apply_softmax": True}
        focal_params = focal_params or {}
        
        # Inicializamos LovaszSoftmax y DiceFocal con los parametros indicados
        self.lovasz = LovaszSoftmaxLoss(**lovasz_params)
        self.focal = DiceFocalLoss(**focal_params)
        
        # Si se pide, normalizamos los pesos para que sumen 1
        total = lovasz_weight + focal_weight if normalize_weights else 1.0
        self.lw = lovasz_weight / total
        self.fw = focal_weight / total
        
        # Si queremos loggear las perdidas parciales
        self.log_partial_losses = log_partial_losses

    def forward(self, logits, targets):
        # Calculamos cada perdida por separado
        lovasz_loss = self.lovasz(logits, targets)
        focal_loss = self.focal(logits, targets)
        combined = self.lw * lovasz_loss + self.fw * focal_loss
        
        # Log opcional para monitorizar valores intermedios
        if self.log_partial_losses:
            print(f"Lovasz Loss: {lovasz_loss.item():.4f}, Focal+Dice Loss: {focal_loss.item():.4f}")
        
        return combined