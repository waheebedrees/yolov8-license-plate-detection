import torch
import torch.nn as nn
from typing import List, Dict, Tuple
import cv2
import numpy as np

from models.common import Conv, C2f, SPPF
from models.head import DetectionHead
from utils.general import nms, pad_to, unpad, unpad_xyxy, xywh2xyxy, make_anchors, dist2bbox
from utils.assigner import TaskAlignedAssigner

class BaseModel(nn.Module):
    def __init__(self, n_classes: int = 1, phi: str = "l"):
        super().__init__()
        self.n_classes = n_classes
        self.phi = phi
        depth_dict = {"n": 0.33, "s": 0.33, "m": 0.67, "l": 1.0, "x": 1.33}
        width_dict = {"n": 0.25, "s": 0.50, "m": 0.75, "l": 1.0, "x": 1.25}
        depth, width = depth_dict[phi], width_dict[phi]

        self.in_channels = [int(256 * width), int(512 * width), int(512 * width * 2)]

        # Backbone
        self.stem = Conv(3, int(64 * width), kernel_size=3, stride=2)
        self.dark2 = nn.Sequential(
            Conv(int(64 * width), int(128 * width), kernel_size=3, stride=2),
            C2f(int(128 * width), int(128 * width), n=round(3 * depth), shortcut=True)
        )
        self.dark3 = nn.Sequential(
            Conv(int(128 * width), int(256 * width), kernel_size=3, stride=2),
            C2f(int(256 * width), int(256 * width), n=round(6 * depth), shortcut=True)
        )
        self.dark4 = nn.Sequential(
            Conv(int(256 * width), int(512 * width), kernel_size=3, stride=2),
            C2f(int(512 * width), int(512 * width), n=round(6 * depth), shortcut=True)
        )
        self.dark5 = nn.Sequential(
            Conv(int(512 * width), int(1024 * width), kernel_size=3, stride=2),
            C2f(int(1024 * width), int(1024 * width), n=round(3 * depth), shortcut=True),
            SPPF(int(1024 * width), int(1024 * width), kernel_size=5)
        )

        # Neck
        self.up_sample = nn.Upsample(scale_factor=2, mode="nearest")
        self.neck_conv1 = C2f(int(1536 * width), int(512 * width), n=round(3 * depth), shortcut=False)
        self.neck_conv2 = C2f(int(768 * width), int(256 * width), n=round(3 * depth), shortcut=False)

        self.neck_conv3 = Conv(int(256 * width), int(256 * width), kernel_size=3, stride=2)
        self.neck_conv4 = C2f(int(768 * width), int(512 * width), n=round(3 * depth), shortcut=False)

        self.neck_conv5 = Conv(int(512 * width), int(512 * width), kernel_size=3, stride=2)
        self.neck_conv6 = C2f(int(1536 * width), int(1024 * width), n=round(3 * depth), shortcut=False)

        # Head
        self.head = DetectionHead(n_classes, in_channels=self.in_channels)

    def forward(self, x: torch.Tensor):
        # Backbone
        x = self.stem(x)
        x2 = self.dark2(x)
        x3 = self.dark3(x2)
        x4 = self.dark4(x3)
        x5 = self.dark5(x4)

        # Neck
        # P5
        x_neck = self.neck_conv1(torch.cat([self.up_sample(x5), x4], dim=1))
        # P4
        x_neck2 = self.neck_conv2(torch.cat([self.up_sample(x_neck), x3], dim=1))
        # P3
        x_neck3 = self.neck_conv3(x_neck2)
        x_neck4 = self.neck_conv4(torch.cat([x_neck3, x_neck], dim=1))
        # P6
        x_neck5 = self.neck_conv5(x_neck4)
        x_neck6 = self.neck_conv6(torch.cat([x_neck5, x5], dim=1))

        # Head
        out = self.head([x_neck2, x_neck4, x_neck6])
        return out

    def predict(self, x: torch.Tensor, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        """
        Performs inference on a batch of images.

        Args:
            x (Tensor): Input batch of images (B, C, H, W)
            conf_threshold (float): Confidence threshold for NMS
            iou_threshold (float): IoU threshold for NMS

        Returns:
            List[Tensor]: List of detections for each image in the batch.
                Each detection is (x1, y1, x2, y2, confidence, class_id)
        """
        preds = self.forward(x)
        preds = nms(preds, conf_threshold, iou_threshold)
        return preds

    def predict_image(self, image_path: str, img_size: Tuple[int, int] = (640, 640), conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        """
        Performs inference on a single image file.

        Args:
            image_path (str): Path to the image file.
            img_size (Tuple[int, int]): Target image size for inference.
            conf_threshold (float): Confidence threshold for NMS.
            iou_threshold (float): IoU threshold for NMS.

        Returns:
            Tuple[np.ndarray, List[np.ndarray]]: Original image with detections drawn, and list of detections.
        """
        original_image = cv2.imread(image_path)
        original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        h0, w0 = original_image_rgb.shape[:2]

        # Preprocess image
        image = original_image_rgb
        if h0 > img_size[0] or w0 > img_size[1]:
            ratio = min(img_size[0] / h0, img_size[1] / w0)
            h, w = int(h0 * ratio), int(w0 * ratio)
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)

        image = image.transpose((2, 0, 1))  # (h, w, 3) -> (3, h, w)
        image = torch.from_numpy(image).float() / 255.0

        # Pad image with black bars to desired img_size
        image, pads = pad_to(image, shape=img_size)
        image = image.unsqueeze(0) # Add batch dimension

        # Perform inference
        with torch.no_grad():
            detections = self.predict(image, conf_threshold, iou_threshold)[0]

        # Scale bounding boxes back to original image size
        if detections.shape[0] > 0:
            detections[:, :4] = unpad_xyxy(detections[:, :4], pads)
            detections[:, :4] = detections[:, :4] / torch.tensor([w, h, w, h], device=detections.device) * torch.tensor([w0, h0, w0, h0], device=detections.device)
            detections = detections.cpu().numpy()

            # Draw bounding boxes on the original image
            for *xyxy, conf, cls in detections:
                x1, y1, x2, y2 = map(int, xyxy)
                label = f'{self.names[int(cls)]} {conf:.2f}'
                cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(original_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return original_image, detections

    def load(self, weights_path: str, device: torch.device):
        """
        Loads model weights from a .pt file.
        """
        self.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        self.to(device)
        self.eval()

    def save(self, path: str):
        """
        Saves model weights to a .pt file.
        """
        torch.save({"model": self}, path)

class DetectionModel(BaseModel):
    def __init__(self, n_classes: int = 1, phi: str = "l"):
        super().__init__(n_classes, phi)
        self.names = {i: str(i) for i in range(n_classes)}
        self.assigner = TaskAlignedAssigner(n_classes=n_classes)

    def set_class_names(self, names: Dict[int, str]):
        self.names = names

    def loss(self, preds: List[torch.Tensor], batch: Dict[str, torch.Tensor], bbox_loss_fn: nn.Module, cls_loss_fn: nn.Module, device: torch.device):
        """
        Calculates the total loss for a batch.

        Args:
            preds (List[Tensor]): Raw predictions from the model.
            batch (Dict): Dictionary containing batched images, labels, etc.
            bbox_loss_fn (nn.Module): Bounding box loss function.
            cls_loss_fn (nn.Module): Classification loss function.
            device (torch.device): Device to perform calculations on.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Total loss, bounding box loss, classification loss.
        """
        total_loss = torch.zeros(1, device=device)
        bbox_loss = torch.zeros(1, device=device)
        cls_loss = torch.zeros(1, device=device)

        # Prepare targets
        gt_classes = batch["cls"].to(device)
        gt_bboxes = batch["bboxes"].to(device)
        batch_idx = batch["batch_idx"].to(device)

        # Get anchor points and strides for all feature maps
        feats = [p.permute(0, 2, 3, 1).contiguous().view(p.shape[0], -1, p.shape[1]) for p in preds]
        anchor_points, stride_tensor = make_anchors(feats, self.head.stride)

        # Concatenate predictions from all feature maps
        # (batch, 4*reg_max + nc, n_layers*height*width)
        preds_cat = torch.cat([p.view(p.shape[0], p.shape[1], -1) for p in preds], dim=2)

        # Split into box and class predictions
        # (batch, 4*reg_max, n_anchors), (batch, nc, n_anchors)
        pred_box_dist, pred_cls = preds_cat.split((4 * self.head.reg_max, self.nc), dim=1)

        # Assign targets
        target_bboxes, target_classes, target_scores, fg_mask, num_pos = self.assigner(
            torch.cat((pred_box_dist, pred_cls), dim=1),
            gt_bboxes, gt_classes, anchor_points, stride_tensor
        )

        if num_pos == 0:
            return total_loss, bbox_loss, cls_loss

        # Calculate bounding box loss
        # Select positive samples
        pred_box_dist_pos = pred_box_dist.permute(0, 2, 1)[fg_mask]
        anchor_points_pos = anchor_points.unsqueeze(0).expand(batch_idx.max() + 1, -1, -1)[fg_mask]
        stride_tensor_pos = stride_tensor.unsqueeze(0).expand(batch_idx.max() + 1, -1, -1)[fg_mask]

        # Convert predicted distances to bounding boxes
        pred_bboxes_pos = dist2bbox(self.head.dfl(pred_box_dist_pos), anchor_points_pos, dim=1) * stride_tensor_pos

        # Calculate IoU loss
        iou = bbox_loss_fn(pred_bboxes_pos, target_bboxes[fg_mask])
        bbox_loss = (iou * target_scores[fg_mask]).sum() / num_pos

        # Calculate classification loss
        pred_cls_pos = pred_cls.permute(0, 2, 1)[fg_mask]
        cls_loss = cls_loss_fn(pred_cls_pos, target_classes[fg_mask])

        total_loss = bbox_loss + cls_loss

        return total_loss, bbox_loss, cls_loss


