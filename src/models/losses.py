import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    Dice loss for binary segmentation.

    This loss expects raw logits from the model and applies sigmoid internally.
    """

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice loss between model logits and binary target masks.
        """
        # Convert logits to probabilities in the range [0, 1].
        probabilities = torch.sigmoid(logits)

        # Flatten tensors so Dice is computed over the full batch.
        probabilities = probabilities.view(-1)
        targets = targets.view(-1)

        intersection = (probabilities * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probabilities.sum() + targets.sum() + self.smooth
        )

        return 1.0 - dice


class BCEDiceLoss(nn.Module):
    """
    Combined BCEWithLogits loss and Dice loss for binary segmentation.
    """

    def __init__(self) -> None:
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute BCE loss plus Dice loss.
        """
        bce_loss = self.bce_loss(logits, targets)
        dice_loss = self.dice_loss(logits, targets)

        return bce_loss + dice_loss


if __name__ == "__main__":
    logits = torch.randn(4, 1, 512, 512)
    targets = torch.randint(0, 2, (4, 1, 512, 512)).float()

    dice_loss_fn = DiceLoss()
    bce_dice_loss_fn = BCEDiceLoss()

    dice_loss = dice_loss_fn(logits, targets)
    bce_dice_loss = bce_dice_loss_fn(logits, targets)

    print(f"Dice Loss: {dice_loss.item():.3f}")
    print(f"BCE + Dice Loss: {bce_dice_loss.item():.3f}")
