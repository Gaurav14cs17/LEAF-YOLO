"""
Train LEAF-YOLO model - Professional Implementation with Adaptive Configuration
"""

import argparse
from leafyolo import LEAFYOLO


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train LEAF-YOLO model')
    parser.add_argument('--task', default='detect', choices=['detect', 'segment', 'classify'], 
                       help='task type')
    parser.add_argument('--model', default='detect', help='model specification (task name, config file, or weights)')
    parser.add_argument('--variant', default='leafyolo_m', 
                       choices=['leafyolo_n', 'leafyolo_s', 'leafyolo_m', 'leafyolo_l', 'leafyolo_x'],
                       help='model variant')
    parser.add_argument('--data', default='coco', help='dataset name or config file')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size')
    parser.add_argument('--img-size', type=int, default=640, help='image size')
    parser.add_argument('--device', default='', help='device')
    parser.add_argument('--project', default='runs/train', help='project name')
    parser.add_argument('--name', default='exp', help='experiment name')
    parser.add_argument('--resume', action='store_true', help='resume training')
    parser.add_argument('--nc', type=int, help='number of classes (overrides dataset config)')
    
    args = parser.parse_args()
    
    # Initialize model with adaptive configuration
    model = LEAFYOLO(
        model=args.model, 
        task=args.task, 
        variant=args.variant,
        nc=args.nc
    )
    
    # Train
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        device=args.device,
        project=args.project,
        name=args.name,
        resume=args.resume
    )
    
    print(f"Training completed! Best model: {results.get('best', 'N/A')}")
    print(f"Last model: {results.get('last', 'N/A')}")


if __name__ == '__main__':
    main()
