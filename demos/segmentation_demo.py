"""
Run YOLO segmentation model on live video feed
"""

from ultralytics import YOLO
import argparse
import cv2
import numpy as np

# Define available models and their paths
models_to_paths = {
    '1': '/Users/sebnico/Desktop/CIS4900/injection-tracker/weights/120-epochs-Oct-8/best.pt',
    '2': '/Users/sebnico/Desktop/CIS4900/injection-tracker/weights/second_dataset/640/best_train6.pt',
    '3': '/Users/sebnico/Desktop/CIS4900/injection-tracker/weights/second_dataset/1280/best_train12.pt',
    '4': '/Users/sebnico/Desktop/CIS4900/injection-tracker/weights/second_dataset/2560/best_train15.pt',
}


def parse_arguments():
    """Parse command line arguments with argparse."""
    parser = argparse.ArgumentParser(
        description='Run YOLO segmentation model on live video feed',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python segmentation_demo.py 1              # Run model 1 only
  python segmentation_demo.py 1 --compare 2  # Compare model 1 and 2 side by side
  python segmentation_demo.py 2 --compare 3   # Compare model 2 and 3 side by side

Available models:
  1 - 120-epochs-Oct-8 model
  2 - second_dataset/640 model
  3 - second_dataset/1280 model
  4 - second_dataset/2560 model
        """
    )
    
    parser.add_argument(
        'model',
        type=str,
        choices=['1', '2', '3', '4'],
        help='Primary model number to use (1-4)'
    )
    
    parser.add_argument(
        '--compare',
        type=str,
        choices=['1', '2', '3', '4'],
        metavar='MODEL_NUM',
        help='Optional second model number to compare side by side'
    )
    
    parser.add_argument(
        '--source',
        type=int,
        default=1,
        help='Video source (default: 1 for webcam)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='mps',
        choices=['cpu', 'cuda', 'mps'],
        help='Device to run inference on (default: mps)'
    )
    
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold for detections (default: 0.25)'
    )
    
    return parser.parse_args()


def main():
    """Main function to run segmentation demo."""
    args = parse_arguments()
    
    # Validate model paths exist
    primary_path = models_to_paths[args.model]
    if args.compare:
        compare_path = models_to_paths[args.compare]
        if args.model == args.compare:
            print("Error: Cannot compare a model with itself!")
            return
    
    if args.compare:
        # Both models side by side
        print(f"Comparing Model {args.model} (left) vs Model {args.compare} (right)")
        first_model = YOLO(primary_path)
        second_model = YOLO(compare_path)
        
        results1 = first_model.track(
            source=args.source, 
            show=False, 
            stream=True, 
            device=args.device,
            conf=args.conf
        )
        results2 = second_model.track(
            source=args.source, 
            show=False, 
            stream=True, 
            device=args.device,
            conf=args.conf
        )
        
        for result1, result2 in zip(results1, results2):
            # Get annotated frames
            frame1 = result1.plot()
            frame2 = result2.plot()
            
            # Combine frames side by side
            combined_frame = np.hstack([frame1, frame2])
            
            # Display combined frame
            cv2.imshow(f'Model {args.model} (Left) | Model {args.compare} (Right)', combined_frame)
            
            # Break on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
    else:
        # Single model only
        print(f"Running Model {args.model} only")
        model = YOLO(primary_path)
        results = model.track(
            source=args.source, 
            show=True, 
            stream=True, 
            device=args.device,
            conf=args.conf
        )
        for r in results:
            pass


if __name__ == '__main__':
    main()
