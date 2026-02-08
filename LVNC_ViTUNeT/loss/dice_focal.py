import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, smooth=1.0, exclude_background=True):
        super().__init__()
        self.gamma = gamma  # factor que controla el enfoque en ejemplos dificiles para focal loss
        self.alpha = alpha  # peso de la clase positiva en focal loss (puede ser tensor por clase)
        self.smooth = smooth  # termino de suavizado para evitar division por cero en dice loss
        self.exclude_background = exclude_background  # si es True, no se incluye la clase fondo en el calculo de dice

    def forward(self, logits, targets):
        num_classes = logits.shape[1]

        # aplicar softmax para obtener probabilidades por clase
        probs = F.softmax(logits, dim=1)

        # convertir etiquetas a one-hot [B, C, H, W]
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        # ----- FOCAL LOSS -----
        # pt es la probabilidad predicha para la clase correcta en cada pixel
        # Clip pt to avoid log(0)
        pt = torch.sum(probs * targets_one_hot, dim=1)  # [B, H, W]
        pt = torch.clamp(pt, min=1e-6, max=1.0)

        # calcular focal loss pixel a pixel
        focal_loss = -self.alpha * ((1 - pt) ** self.gamma) * torch.log(pt + 1e-8)
        focal_loss = focal_loss.mean()  # promedio sobre batch y espacial

        # ----- DICE LOSS -----
        # calcular interseccion y union por clase
        intersection = torch.sum(probs * targets_one_hot, dim=(0, 2, 3))  # [C]
        union = torch.sum(probs + targets_one_hot, dim=(0, 2, 3))  # [C]
        # calcular dice score por clase
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_score  # convertir a perdida

        if self.exclude_background:
            dice_loss = dice_loss[1:]  # excluir la clase fondo (indice 0) si se desea

        dice_loss = dice_loss.mean()  # promedio sobre clases

        # retornar la combinacion de ambas perdidas
        return focal_loss + dice_loss
