import torch
import yaml
import os
import cv2
import u
from models.yolo import DetectionModel


def predict(image_path: str, weights_path: str, config_path: str = "config/model_config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model
    model = DetectionModel(n_classes=len(
        config["dataset"]["names"]), phi=config["model"]["phi"])
    model.set_class_names(
        names={i: name for i, name in enumerate(config["dataset"]["names"])})
    model.load(weights_path, device)
    model.eval()

    # Perform prediction
    output_image, detections = model.predict_image(
        image_path,
        img_size=config["train"]["img_size"],
        conf_threshold=config["train"]["confidence_thresh"],
        iou_threshold=config["train"]["iou_thresh"]
    )

    # Save or display output_image
    output_dir = "inference_results"
    os.makedirs(output_dir, exist_ok=True)
    output_image_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_image_path, output_image)
    print(f"Detections saved to {output_image_path}")

    # Print detections
    if len(detections) > 0:
        print("Detections:")
        for *xyxy, conf, cls in detections:
            print(
                f"  Class: {model.names[int(cls)]}, Confidence: {conf:.2f}, Bbox: [{int(xyxy[0])}, {int(xyxy[1])}, {int(xyxy[2])}, {int(xyxy[3])}]")
    else:
        print("No detections found.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="YOLOv8 License Plate Detection Prediction")
    parser.add_argument("--image_path", type=str,
                        required=True, help="Path to the input image.")
    parser.add_argument("--weights_path", type=str, required=True,
                        help="Path to the trained model weights (.pt file).")
    parser.add_argument("--config_path", type=str, default="config/model_config.yaml",
                        help="Path to the model configuration YAML file.")

    args = parser.parse_args()

    predict(args.image_path, args.weights_path, args.config_path)
