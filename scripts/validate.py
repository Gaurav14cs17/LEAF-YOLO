"""
Validate LEAF-YOLO model - Professional Implementation
"""

import argparse
from leafyolo import LEAFYOLO


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(description='LEAF-YOLO Validation')
    parser.add_argument('--model', default='leafyolo.pt', help='model path')
    parser.add_argument('--data', default='data/coco.yaml', help='dataset configuration')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--img-size', type=int, default=640, help='image size')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IoU threshold')
    parser.add_argument('--device', default='', help='device')
    parser.add_argument('--project', default='runs/val', help='project name')
    parser.add_argument('--name', default='exp', help='experiment name')
    parser.add_argument('--save-json', action='store_true', help='save results to JSON')
    parser.add_argument('--save-txt', action='store_true', help='save results to txt')
    
    args = parser.parse_args()
    
    # Initialize model
    model = LEAFYOLO(args.model)
    
    # Validate
    results = model.val(
        data=args.data,
        batch_size=args.batch_size,
        img_size=args.img_size,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        device=args.device,
        project=args.project,
        name=args.name,
        save_json=args.save_json,
        save_txt=args.save_txt
    )
    
    print(f"Validation completed!")
    print(f"mAP@0.5: {results['map50']:.3f}")
    print(f"mAP@0.5:0.95: {results['map']:.3f}")


if __name__ == '__main__':
    main()
