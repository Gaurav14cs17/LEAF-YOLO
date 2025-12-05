#!/usr/bin/env python3
"""
LEAF-YOLO CLI - Professional Implementation with Adaptive Configuration
Simple command-line interface for all LEAF-YOLO tasks
"""

import argparse
from leafyolo import LEAFYOLO
from leafyolo.utils.config import config_manager


def train(args):
    """Training command."""
    model = LEAFYOLO(model=args.task, variant=args.variant, nc=args.nc)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        device=args.device,
        project=args.project,
        name=args.name
    )
    print(f"✅ Training completed! Best: {results.get('best', 'N/A')}")


def predict(args):
    """Prediction command."""
    model = LEAFYOLO(model=args.model, task=args.task, variant=args.variant, nc=args.nc)
    results = model.predict(
        source=args.source,
        img_size=args.img_size,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        device=args.device,
        save=args.save
    )
    print(f"✅ Prediction completed! {len(results)} images processed")


def validate(args):
    """Validation command."""
    model = LEAFYOLO(model=args.model, task=args.task, variant=args.variant)
    results = model.val(
        data=args.data,
        batch_size=args.batch_size,
        img_size=args.img_size,
        device=args.device
    )
    print(f"✅ Validation completed!")
    print(f"   mAP@0.5: {results.get('map50', 0):.3f}")
    print(f"   mAP@0.5:0.95: {results.get('map', 0):.3f}")


def export(args):
    """Export command."""
    model = LEAFYOLO(model=args.model, task=args.task, variant=args.variant)
    exported_path = model.export(
        format=args.format,
        img_size=args.img_size,
        dynamic=args.dynamic,
        simplify=args.simplify
    )
    print(f"✅ Model exported to: {exported_path}")


def info(args):
    """Show model/config information."""
    if args.config_info:
        print("🔧 LEAF-YOLO Configuration System")
        print(f"Available tasks: {config_manager.list_tasks()}")
        print(f"Available variants: {config_manager.list_variants()}")
        print(f"Available datasets: {config_manager.list_datasets()}")
    else:
        model = LEAFYOLO(model=args.model, task=args.task, variant=args.variant)
        model.info(detailed=args.detailed)


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description='LEAF-YOLO CLI - Professional Implementation')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Common arguments
    def add_common_args(p):
        p.add_argument('--task', default='detect', choices=['detect', 'segment', 'classify'])
        p.add_argument('--variant', default='leafyolo_m', 
                      choices=['leafyolo_n', 'leafyolo_s', 'leafyolo_m', 'leafyolo_l', 'leafyolo_x'])
        p.add_argument('--device', default='', help='device (cpu, cuda:0, etc.)')
        p.add_argument('--nc', type=int, help='number of classes')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train model')
    add_common_args(train_parser)
    train_parser.add_argument('--data', default='coco', help='dataset')
    train_parser.add_argument('--epochs', type=int, default=100)
    train_parser.add_argument('--batch-size', type=int, default=16)
    train_parser.add_argument('--img-size', type=int, default=640)
    train_parser.add_argument('--project', default='runs/train')
    train_parser.add_argument('--name', default='exp')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Run prediction')
    add_common_args(predict_parser)
    predict_parser.add_argument('--model', default='leafyolo.pt', help='model path')
    predict_parser.add_argument('--source', required=True, help='source images/video')
    predict_parser.add_argument('--img-size', type=int, default=640)
    predict_parser.add_argument('--conf-thres', type=float, default=0.25)
    predict_parser.add_argument('--iou-thres', type=float, default=0.45)
    predict_parser.add_argument('--save', action='store_true', help='save results')
    
    # Validate command
    val_parser = subparsers.add_parser('val', help='Validate model')
    add_common_args(val_parser)
    val_parser.add_argument('--model', default='leafyolo.pt', help='model path')
    val_parser.add_argument('--data', default='coco', help='dataset')
    val_parser.add_argument('--batch-size', type=int, default=32)
    val_parser.add_argument('--img-size', type=int, default=640)
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export model')
    add_common_args(export_parser)
    export_parser.add_argument('--model', default='leafyolo.pt', help='model path')
    export_parser.add_argument('--format', default='onnx', 
                              choices=['onnx', 'tensorrt', 'coreml', 'tflite'])
    export_parser.add_argument('--img-size', type=int, default=640)
    export_parser.add_argument('--dynamic', action='store_true', help='dynamic axes')
    export_parser.add_argument('--simplify', action='store_true', help='simplify onnx')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show model info')
    add_common_args(info_parser)
    info_parser.add_argument('--model', default='detect', help='model specification')
    info_parser.add_argument('--detailed', action='store_true', help='detailed info')
    info_parser.add_argument('--config-info', action='store_true', help='show config system info')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        # Execute command
        if args.command == 'train':
            train(args)
        elif args.command == 'predict':
            predict(args)
        elif args.command == 'val':
            validate(args)
        elif args.command == 'export':
            export(args)
        elif args.command == 'info':
            info(args)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
