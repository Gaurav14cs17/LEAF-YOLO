"""
Predict with LEAF-YOLO model - Professional Implementation with Adaptive Configuration
"""

import argparse
from leafyolo import LEAFYOLO


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description='LEAF-YOLO Prediction')
    parser.add_argument('--task', default='detect', choices=['detect', 'segment', 'classify'], 
                       help='task type')
    parser.add_argument('--model', default='leafyolo.pt', help='model path or task name')
    parser.add_argument('--variant', default='leafyolo_m', 
                       choices=['leafyolo_n', 'leafyolo_s', 'leafyolo_m', 'leafyolo_l', 'leafyolo_x'],
                       help='model variant (if using task name)')
    parser.add_argument('--source', default='data/images', help='source')
    parser.add_argument('--img-size', type=int, default=640, help='image size')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IoU threshold')
    parser.add_argument('--device', default='', help='device')
    parser.add_argument('--project', default='runs/predict', help='project name')
    parser.add_argument('--name', default='exp', help='experiment name')
    parser.add_argument('--save', action='store_true', help='save results')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--nc', type=int, help='number of classes (if using task name)')
    
    args = parser.parse_args()
    
    # Initialize model with adaptive configuration
    model = LEAFYOLO(
        model=args.model, 
        task=args.task, 
        variant=args.variant,
        nc=args.nc
    )
    
    # Predict
    results = model.predict(
        source=args.source,
        img_size=args.img_size,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        device=args.device,
        project=args.project,
        name=args.name,
        save=args.save,
        view_img=args.view_img
    )
    
    print(f"Prediction completed! {len(results)} images processed")
    
    # Show some results info
    for i, result in enumerate(results[:3]):  # Show first 3 results
        detections = result.get('detections', [])
        print(f"Image {i+1}: {len(detections)} detections found")


if __name__ == '__main__':
    main()
