import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import os
from tqdm import tqdm

from models.yolo import DetectionModel
from utils.loss import DetectionLoss
from data.dataset import Dataset

def train(config_path: str = "config/model_config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset and DataLoader
    train_dataset = Dataset(config["dataset"], mode="train", img_size=config["train"]["img_size"])
    train_dataloader = DataLoader(train_dataset, batch_size=config["train"]["batch_size"], shuffle=True, collate_fn=train_dataset.collate_fn)

    val_dataset = Dataset(config["dataset"], mode="val", img_size=config["train"]["img_size"])
    val_dataloader = DataLoader(val_dataset, batch_size=config["train"]["batch_size"], shuffle=False, collate_fn=val_dataset.collate_fn)

    # Model
    model = DetectionModel(n_classes=len(config["dataset"]["names"]), phi=config["model"]["phi"])
    model.set_class_names(names={i: name for i, name in enumerate(config["dataset"]["names"])})
    model.to(device)

    # Loss function
    criterion = DetectionLoss(
        n_classes=len(config["dataset"]["names"]),
        iou_mode=config["train"]["iou_mode"],
        reg_max=config["train"]["reg_max"],
        bbox_weight=config["train"]["bbox_weight"],
        cls_weight=config["train"]["cls_weight"],
        dfl_weight=config["train"]["dfl_weight"]
    )

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["train"]["learning_rate"],
        weight_decay=config["train"]["weight_decay"]
    )

    # Training loop
    epochs = config["train"]["epochs"]
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        bbox_loss_sum = 0.0
        cls_loss_sum = 0.0

        print(f"Epoch {epoch+1}/{epochs}")
        for batch in tqdm(train_dataloader):
            preds = model(batch["images"].to(device))
            loss, bbox_l, cls_l = model.loss(preds, batch, criterion.bbox_loss, criterion.cls_loss, device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            bbox_loss_sum += bbox_l.item()
            cls_loss_sum += cls_l.item()

        avg_total_loss = total_loss / len(train_dataloader)
        avg_bbox_loss = bbox_loss_sum / len(train_dataloader)
        avg_cls_loss = cls_loss_sum / len(train_dataloader)

        print(f"Train Loss: {avg_total_loss:.4f}, Bbox Loss: {avg_bbox_loss:.4f}, Cls Loss: {avg_cls_loss:.4f}")

        # Validation loop
        model.eval()
        val_total_loss = 0.0
        val_bbox_loss_sum = 0.0
        val_cls_loss_sum = 0.0
        with torch.no_grad():
            for batch in tqdm(val_dataloader):
                preds = model(batch["images"].to(device))
                loss, bbox_l, cls_l = model.loss(preds, batch, criterion.bbox_loss, criterion.cls_loss, device)

                val_total_loss += loss.item()
                val_bbox_loss_sum += bbox_l.item()
                val_cls_loss_sum += cls_l.item()

        avg_val_total_loss = val_total_loss / len(val_dataloader)
        avg_val_bbox_loss = val_bbox_loss_sum / len(val_dataloader)
        avg_val_cls_loss = val_cls_loss_sum / len(val_dataloader)

        print(f"Val Loss: {avg_val_total_loss:.4f}, Val Bbox Loss: {avg_val_bbox_loss:.4f}, Val Cls Loss: {avg_val_cls_loss:.4f}")

        # Save model checkpoint
        os.makedirs("weights", exist_ok=True)
        model.save(os.path.join("weights", f"yolov8_epoch_{epoch+1}.pt"))

if __name__ == "__main__":
    train()


