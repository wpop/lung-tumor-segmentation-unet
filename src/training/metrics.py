import torch


def _prepare_binary_tensors(
    pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert prediction and target tensors to flattened binary tensors.
    """
    pred_binary = (pred > threshold).float().view(-1)
    target_binary = (target > threshold).float().view(-1)

    return pred_binary, target_binary


def dice_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute the Dice score for binary segmentation predictions.
    """
    pred_binary, target_binary = _prepare_binary_tensors(pred, target)
    smooth = 1e-7

    intersection = (pred_binary * target_binary).sum()
    score = (2.0 * intersection + smooth) / (
        pred_binary.sum() + target_binary.sum() + smooth
    )

    return float(score.item())


def iou_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute the Intersection over Union score for binary segmentation predictions.
    """
    pred_binary, target_binary = _prepare_binary_tensors(pred, target)
    smooth = 1e-7

    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection
    score = (intersection + smooth) / (union + smooth)

    return float(score.item())


def precision_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute the precision score for binary segmentation predictions.
    """
    pred_binary, target_binary = _prepare_binary_tensors(pred, target)
    smooth = 1e-7

    true_positive = (pred_binary * target_binary).sum()
    predicted_positive = pred_binary.sum()
    score = true_positive / (predicted_positive + smooth)

    return float(score.item())


def recall_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute the recall score for binary segmentation predictions.
    """
    pred_binary, target_binary = _prepare_binary_tensors(pred, target)
    smooth = 1e-7

    true_positive = (pred_binary * target_binary).sum()
    actual_positive = target_binary.sum()
    score = true_positive / (actual_positive + smooth)

    return float(score.item())


if __name__ == "__main__":
    predictions = torch.rand(2, 1, 64, 64)
    targets = torch.randint(0, 2, (2, 1, 64, 64)).float()

    print(f"Dice score: {dice_score(predictions, targets):.4f}")
    print(f"IoU score: {iou_score(predictions, targets):.4f}")
    print(f"Precision score: {precision_score(predictions, targets):.4f}")
    print(f"Recall score: {recall_score(predictions, targets):.4f}")
