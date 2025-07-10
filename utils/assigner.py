import torch
import torch.nn as nn

from utils.general import bbox_iou, anchors_in_gt_boxes, dist2bbox, select_highest_iou

class TaskAlignedAssigner(nn.Module):
    """
    Task-Aligned Assigner for YOLOv8

    Assigns ground truth boxes to predicted boxes based on alignment metrics
    (IoU and classification scores).
    """
    def __init__(self, topk: int = 13, n_classes: int = 1):
        super().__init__()
        self.topk = topk
        self.n_classes = n_classes

    @torch.no_grad()
    def forward(self, preds: torch.Tensor, gt_bboxes: torch.Tensor, gt_classes: torch.Tensor, anchor_points: torch.Tensor, stride_tensor: torch.Tensor):
        """
        Args:
            preds (Tensor): Predictions from model (batch_size, 4 + n_classes, n_anchors)
            gt_bboxes (Tensor): Ground truth bounding boxes (batch_size, n_gt_boxes, 4)
            gt_classes (Tensor): Ground truth classes (batch_size, n_gt_boxes)
            anchor_points (Tensor): Anchor points (n_anchors, 2)
            stride_tensor (Tensor): Strides for each anchor point (n_anchors, 1)

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
                target_bboxes (Tensor): Assigned ground truth bounding boxes (batch_size, n_anchors, 4)
                target_classes (Tensor): Assigned ground truth classes (batch_size, n_anchors)
                target_scores (Tensor): Assigned scores (batch_size, n_anchors)
                fg_mask (Tensor): Foreground mask (batch_size, n_anchors)
                num_pos (int): Number of positive samples
        """
        batch_size, n_gt_boxes, _ = gt_bboxes.shape
        n_anchors = anchor_points.shape[0]

        # (batch_size, 4, n_anchors), (batch_size, n_classes, n_anchors)
        pred_bboxes, pred_classes = preds.split((4, self.n_classes), dim=1)

        # Convert pred_bboxes from ltrb to xyxy
        pred_bboxes = dist2bbox(pred_bboxes, anchor_points.unsqueeze(0), xywh=False)

        # Reshape for broadcasting
        # (batch_size, n_anchors, 4) -> (batch_size, n_anchors, 1, 4)
        pred_bboxes = pred_bboxes.permute(0, 2, 1).unsqueeze(2)
        # (batch_size, n_classes, n_anchors) -> (batch_size, n_anchors, n_classes)
        pred_classes = pred_classes.permute(0, 2, 1)
        # (batch_size, n_gt_boxes, 4) -> (batch_size, 1, n_gt_boxes, 4)
        gt_bboxes = gt_bboxes.unsqueeze(1)
        # (batch_size, n_gt_boxes) -> (batch_size, 1, n_gt_boxes)
        gt_classes = gt_classes.unsqueeze(1)

        # Calculate IoU between predicted and ground truth boxes
        # (batch_size, n_anchors, n_gt_boxes)
        ious = bbox_iou(pred_bboxes, gt_bboxes, xywh=False).squeeze(-1)

        # Calculate alignment metric
        # (batch_size, n_anchors, n_gt_boxes)
        alignment_metric = ious.pow(0.5) * pred_classes.softmax(dim=-1).gather(dim=-1, index=gt_classes.expand(-1, n_anchors, -1).long()).pow(0.5)

        # Select top-k anchors for each GT box
        # (batch_size, n_gt_boxes, topk)
        topk_metric, topk_idx = torch.topk(alignment_metric, self.topk, dim=1, largest=True)

        # Create mask for top-k anchors
        # (batch_size, n_anchors, n_gt_boxes)
        mask = torch.zeros_like(alignment_metric, dtype=torch.bool)
        # Scatter top-k indices to create the mask
        mask.scatter_(dim=1, index=topk_idx, src=torch.ones_like(topk_metric, dtype=torch.bool))

        # Filter out anchors that are not within their corresponding GT box
        # (batch_size, n_anchors, n_gt_boxes)
        in_gt_mask = anchors_in_gt_boxes(anchor_points, gt_bboxes.squeeze(1))
        mask = mask * in_gt_mask

        # Select the best GT box for each anchor if multiple GT boxes are assigned
        # (batch_size, n_anchors)
        target_gt_box_idxs, fg_mask, mask = select_highest_iou(mask, ious, n_gt_boxes)

        # Prepare targets
        target_bboxes = torch.zeros(batch_size, n_anchors, 4, device=preds.device)
        target_classes = torch.zeros(batch_size, n_anchors, dtype=torch.long, device=preds.device)
        target_scores = torch.zeros(batch_size, n_anchors, device=preds.device)

        for i in range(batch_size):
            if fg_mask[i].sum() > 0:
                target_bboxes[i, fg_mask[i]] = gt_bboxes[i, 0, target_gt_box_idxs[i, fg_mask[i]]]
                target_classes[i, fg_mask[i]] = gt_classes[i, 0, target_gt_box_idxs[i, fg_mask[i]]]
                target_scores[i, fg_mask[i]] = alignment_metric[i, fg_mask[i], target_gt_box_idxs[i, fg_mask[i]]]

        num_pos = fg_mask.sum().item()

        return target_bboxes, target_classes, target_scores, fg_mask, num_pos


