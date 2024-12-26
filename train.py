import argparse
import logging
import os
import yaml
from datetime import datetime
from pathlib import Path

import torch
from ultralytics import YOLO

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_logging(save_dir):
    """Setup logging configuration"""
    log_file = save_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

def load_config(config_path):
    """Load configuration file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_yolov8(data_yaml, 
                 weights=None,
                 epochs=100,
                 batch_size=16,
                 img_size=640,
                 device='cuda',
                 project='runs/train',
                 name='yolov8_exp'):
    """
    Train YOLOv8 model
    """
    logger.info("Starting YOLOv8 training...")
    
    # Initialize model
    if weights:
        model = YOLO(weights)
        logger.info(f"Loaded weights from: {weights}")
    else:
        model = YOLO('yolov8x.pt')  # Use pretrained YOLOv8x
        logger.info("Using pretrained YOLOv8x weights")
    
    # Train the model
    try:
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            device=device,
            project=project,
            name=name
        )
        logger.info("YOLOv8 training completed successfully")
        return results
    except Exception as e:
        logger.error(f"Error during YOLOv8 training: {str(e)}")
        raise

def train_yolov5(data_yaml,
                 weights=None,
                 epochs=100,
                 batch_size=16,
                 img_size=640,
                 device='cuda',
                 project='runs/train',
                 name='yolov5_exp'):
    """
    Train YOLOv5 model
    """
    logger.info("Starting YOLOv5 training...")
    
    try:
        import yolov5
    except ImportError:
        logger.error("YOLOv5 package not found. Install it using: pip install yolov5")
        raise
    
    # Initialize model
    if weights:
        model = yolov5.load(weights)
        logger.info(f"Loaded weights from: {weights}")
    else:
        model = yolov5.load('yolov5s.pt')
        logger.info("Using pretrained YOLOv5s weights")
    
    # Train the model
    try:
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            batch_size=batch_size,
            img_size=img_size,
            device=device,
            project=project,
            name=name
        )
        logger.info("YOLOv5 training completed successfully")
        return results
    except Exception as e:
        logger.error(f"Error during YOLOv5 training: {str(e)}")
        raise

def train_yolov11(data_yaml,
                  weights=None,
                  epochs=100,
                  batch_size=16,
                  img_size=640,
                  device='cuda',
                  project='runs/train',
                  name='yolov11_exp'):
    """
    Train YOLOv11n model
    Note: Implement according to YOLOv11 specifications when available
    """
    logger.info("Starting YOLOv11 training...")
    raise NotImplementedError("YOLOv11 training not yet implemented")

def main():
    parser = argparse.ArgumentParser(description='Train YOLO models')
    parser.add_argument('--model', type=str, required=True, choices=['yolov5', 'yolov8', 'yolov11'],
                      help='Model version to train')
    parser.add_argument('--data', type=str, required=True,
                      help='Path to data.yaml file')
    parser.add_argument('--weights', type=str, default=None,
                      help='Path to pretrained weights')
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=16,
                      help='Batch size')
    parser.add_argument('--img-size', type=int, default=640,
                      help='Image size')
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to train on (cuda/cpu)')
    parser.add_argument('--project', type=str, default='runs/train',
                      help='Project name for saving results')
    parser.add_argument('--name', type=str, default=None,
                      help='Experiment name')
    
    args = parser.parse_args()
    
    # Setup save directory
    save_dir = Path(args.project) / (args.name or datetime.now().strftime("%Y%m%d_%H%M%S"))
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logging(save_dir)
    
    # Log training configuration
    logger.info("Training configuration:")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
    
    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU instead")
        args.device = 'cpu'
    
    # Save configuration
    config_path = save_dir / 'train_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(vars(args), f)
    
    try:
        # Train model based on version
        if args.model == 'yolov8':
            results = train_yolov8(
                args.data,
                args.weights,
                args.epochs,
                args.batch_size,
                args.img_size,
                args.device,
                args.project,
                args.name
            )
        elif args.model == 'yolov5':
            results = train_yolov5(
                args.data,
                args.weights,
                args.epochs,
                args.batch_size,
                args.img_size,
                args.device,
                args.project,
                args.name
            )
        elif args.model == 'yolov11':
            results = train_yolov11(
                args.data,
                args.weights,
                args.epochs,
                args.batch_size,
                args.img_size,
                args.device,
                args.project,
                args.name
            )
        
        logger.info(f"Training completed. Results saved to {save_dir}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()