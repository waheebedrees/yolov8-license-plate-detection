import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.general import bbox_iou, df_loss

class BboxLoss(nn.Module):
    """
    Bounding box loss function.

    Combines IoU loss and Distribution Focal Loss (DFL).
    """
    def __init__(self, iou_mode: str = "ciou", reg_max: int = 16):
        super().__init__()
        assert iou_mode in ("iou", "giou", "diou", "ciou"), f"Invalid IoU mode: {iou_mode}"
        self.iou_mode = iou_mode
        self.reg_max = reg_max

    def forward(self, pred_bboxes: torch.Tensor, target_bboxes: torch.Tensor, pred_box_dist: torch.Tensor = None, target_dfl: torch.Tensor = None):
        """
        Args:
            pred_bboxes (Tensor): Predicted bounding boxes (N, 4)
            target_bboxes (Tensor): Ground truth bounding boxes (N, 4)
            pred_box_dist (Tensor, optional): Predicted box distribution for DFL (N, 4 * reg_max). Required for DFL.
            target_dfl (Tensor, optional): Target for DFL (N, 4). Required for DFL.

        Returns:
            Tensor: Total bounding box loss.
        """
        # IoU Loss
        iou = bbox_iou(pred_bboxes, target_bboxes, xywh=False, iou_mode=self.iou_mode)
        iou_loss = 1.0 - iou

        # DFL Loss
        if pred_box_dist is not None and target_dfl is not None:
            dfl = df_loss(pred_box_dist.view(-1, self.reg_max), target_dfl.view(-1))
            return iou_loss.mean() + dfl.mean()

        return iou_loss.mean()

class DetectionLoss(nn.Module):
    """
    Combines bounding box loss and classification loss.
    """
    def __init__(self, n_classes: int = 1, iou_mode: str = "ciou", reg_max: int = 16, bbox_weight: float = 7.5, cls_weight: float = 0.5, dfl_weight: float = 1.5):
        super().__init__()
        self.n_classes = n_classes
        self.bbox_loss = BboxLoss(iou_mode=iou_mode, reg_max=reg_max)
        self.cls_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.bbox_weight = bbox_weight
        self.cls_weight = cls_weight
        self.dfl_weight = dfl_weight

    def forward(self, pred_bboxes: torch.Tensor, target_bboxes: torch.Tensor, pred_box_dist: torch.Tensor, target_dfl: torch.Tensor, pred_cls: torch.Tensor, target_cls: torch.Tensor, target_scores: torch.Tensor):
        """
        Args:
            pred_bboxes (Tensor): Predicted bounding boxes (N, 4)
            target_bboxes (Tensor): Ground truth bounding boxes (N, 4)
            pred_box_dist (Tensor): Predicted box distribution for DFL (N, 4 * reg_max)
            target_dfl (Tensor): Target for DFL (N, 4)
            pred_cls (Tensor): Predicted class logits (N, n_classes)
            target_cls (Tensor): Ground truth classes (N,)
            target_scores (Tensor): Target scores for classification (N,)

        Returns:
            Tensor: Total detection loss.
        """
        # Bounding Box Loss
        bbox_loss = self.bbox_loss(pred_bboxes, target_bboxes, pred_box_dist, target_dfl) * self.bbox_weight

        # Classification Loss
        target_cls_one_hot = F.one_hot(target_cls, num_classes=self.n_classes).float()
        cls_loss = self.cls_loss(pred_cls, target_cls_one_hot) * target_scores.unsqueeze(-1)
        cls_loss = cls_loss.sum() / target_scores.sum()
        cls_loss *= self.cls_weight

        return bbox_loss + cls_loss


