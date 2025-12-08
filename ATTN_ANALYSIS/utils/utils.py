import torch

def temporal_repeat(tensor):
    """
    Repeat the input tensor along the temporal dimension.
    Args:
        tensor (torch.Tensor): Input tensor of shape (T, H, W).
    Returns:
    torch.Tensor: Repeated tensor of shape (T * N, H, W) where N is the number of repetitions.
    """
    tensor = tensor.unsqueeze(1) # Add a channel dimension
    tensor = tensor.repeat(1, 4, 1, 1) # Repeat along the channel dimension
    tensor = tensor.view(-1, tensor.shape[2], tensor.shape[3]) # Flatten the channel dimension
    return tensor

# FN: COMPUTE IOU: INTERSECTION OVER UNION
def compute_iou(pred_logits, gt_masks, threshold=0.5):
    """
    Compute Intersection over Union (IoU) between predicted logits and ground truth masks.
    Args:
        pred_logits (torch.Tensor): Predicted logits of shape (N, H, W).
        gt_masks (torch.Tensor): Ground truth masks of shape (N, H, W).
        threshold (float): Threshold to binarize the predicted logits.
    Returns:
        iou (torch.Tensor): IoU values of shape (N,).
    """

    pred_masks = (pred_logits > threshold).float()
    intersection = (pred_masks * gt_masks).sum(dim=(1, 2))
    union = (pred_masks + gt_masks).sum(dim=(1, 2)) - intersection
    iou = intersection / union
    iou[union == 0] = 1.0
    return iou.mean()

# FN: COMPUTE CROSS ENTROPY
def compute_binary_cross_entropy(pred_logits, gt_masks):
    """
    Compute binary cross entropy loss between predicted logits and ground truth masks.
    Args:
        pred_logits (torch.Tensor): Predicted logits of shape (N, H, W).
        gt_masks (torch.Tensor): Ground truth masks of shape (N, H, W).
    Returns:
        bce_loss (torch.Tensor): Binary cross entropy loss. (N,)
    """
    bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_logits, gt_masks, reduction='none')
    bce_loss = bce_loss.mean(dim=(1, 2))  # Average over spatial dimensions
    return bce_loss.mean()

# FN: COMPUTE DICE LOSS
def compute_dice(pred_logits, gt_masks):
    """
    Compute Dice loss between predicted logits and ground truth masks.
    Args:
        pred_logits (torch.Tensor): Predicted logits of shape (N, H, W).
        gt_masks (torch.Tensor): Ground truth masks of shape (N, H, W).
    Returns:
        dice_loss (torch.Tensor): Dice loss. (N,)
    """
    pred_masks = torch.sigmoid(pred_logits)
    intersection = (pred_masks * gt_masks).sum(dim=(1, 2))
    union = pred_masks.sum(dim=(1, 2)) + gt_masks.sum(dim=(1, 2))
    dice_loss = 1 - (2 * intersection + 1e-6) / (union + 1e-6)
    return dice_loss.mean()

# FN: COMPUTE FOCAL LOSS
def compute_focal_loss(pred_logits, gt_masks, alpha=0.25, gamma=2.0):
    """
    Compute Focal loss between predicted logits and ground truth masks.
    Args:
        pred_logits (torch.Tensor): Predicted logits of shape (N, H, W).
        gt_masks (torch.Tensor): Ground truth masks of shape (N, H, W).
        alpha (float): Focusing parameter.
        gamma (float): Focusing parameter.
    Returns:
        focal_loss (torch.Tensor): Focal loss. (N,)
    """
    bce_loss = compute_binary_cross_entropy(pred_logits, gt_masks)
    pt = torch.exp(-bce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * bce_loss
    return focal_loss.mean()