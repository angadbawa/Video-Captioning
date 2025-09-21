import argparse
import logging
import sys
import json
from pathlib import Path
import torch
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.inference.predictor import VideoCaptionPredictor, BatchPredictor


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def predict_single_video(args):
    """Predict caption for a single video."""
    logger = logging.getLogger(__name__)
    
    # Initialize predictor
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    predictor = VideoCaptionPredictor(Path(args.model_path), device=device)
    
    # Generate prediction
    logger.info(f"Generating caption for: {args.video_path}")
    
    if args.features_path:
        features = np.load(args.features_path)
        result = predictor.predict_from_features(
            video_features=features,
            method=args.method,
            max_length=args.max_length,
            beam_size=args.beam_size,
            length_penalty=args.length_penalty,
            temperature=args.temperature
        )
    else:
        # Extract features from video
        result = predictor.predict_from_video(
            video_path=Path(args.video_path),
            method=args.method,
            max_length=args.max_length,
            beam_size=args.beam_size,
            length_penalty=args.length_penalty,
            temperature=args.temperature
        )
    
    # Print result
    print(f"\nGenerated Caption: {result['caption']}")
    print(f"Method: {result['method']}")
    print(f"Tokens: {result['tokens']}")
    
    # Save result if output path specified
    if args.output:
        output_data = {
            'video_path': args.video_path,
            'caption': result['caption'],
            'method': args.method,
            'tokens': result['tokens'],
            'parameters': {
                'max_length': args.max_length,
                'beam_size': args.beam_size,
                'length_penalty': args.length_penalty,
                'temperature': args.temperature
            }
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Results saved to: {args.output}")


def predict_batch_videos(args):
    """Predict captions for multiple videos."""
    logger = logging.getLogger(__name__)
    
    # Load video paths
    if args.video_list.endswith('.txt'):
        with open(args.video_list, 'r') as f:
            video_paths = [Path(line.strip()) for line in f if line.strip()]
    else:
        # Assume it's a directory
        video_dir = Path(args.video_list)
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
        video_paths = []
        for ext in video_extensions:
            video_paths.extend(video_dir.glob(f'*{ext}'))
    
    logger.info(f"Found {len(video_paths)} videos to process")
    
    # Initialize predictor
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    predictor = VideoCaptionPredictor(Path(args.model_path), device=device)
    batch_predictor = BatchPredictor(predictor, batch_size=args.batch_size)
    
    # Generate predictions
    logger.info("Starting batch prediction...")
    results = batch_predictor.predict_videos(
        video_paths=video_paths,
        method=args.method,
        max_length=args.max_length,
        beam_size=args.beam_size,
        length_penalty=args.length_penalty,
        temperature=args.temperature
    )
    
    # Print results
    for result in results:
        print(f"\nVideo: {result.get('video_path', 'Unknown')}")
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Caption: {result['caption']}")
    
    # Save results
    if args.output:
        output_data = {
            'parameters': {
                'method': args.method,
                'max_length': args.max_length,
                'beam_size': args.beam_size,
                'length_penalty': args.length_penalty,
                'temperature': args.temperature
            },
            'results': results
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Results saved to: {args.output}")
    
    # Save captions in simple format
    if args.captions_file:
        with open(args.captions_file, 'w') as f:
            for result in results:
                if 'error' not in result:
                    f.write(f"{result['caption']}\n")
                else:
                    f.write("\n")  # Empty line for failed predictions
        
        logger.info(f"Captions saved to: {args.captions_file}")


def predict_multiple_captions(args):
    """Generate multiple diverse captions for a single video."""
    logger = logging.getLogger(__name__)
    
    # Initialize predictor
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    predictor = VideoCaptionPredictor(Path(args.model_path), device=device)
    
    # Load features
    if args.features_path:
        import numpy as np
        features = np.load(args.features_path)
    else:
        # Extract features from video
        features = predictor._extract_video_features(Path(args.video_path))
    
    # Generate multiple captions
    logger.info(f"Generating {args.num_captions} captions for: {args.video_path}")
    
    captions = predictor.generate_multiple_captions(
        video_features=features,
        num_captions=args.num_captions,
        method=args.method,
        max_length=args.max_length,
        beam_size=max(args.beam_size, args.num_captions),
        temperature=args.temperature
    )
    
    # Print results
    print(f"\nGenerated {len(captions)} captions:")
    for i, caption_data in enumerate(captions, 1):
        print(f"{i}. {caption_data['caption']} (score: {caption_data['score']:.3f})")
    
    # Save results
    if args.output:
        output_data = {
            'video_path': args.video_path,
            'captions': captions,
            'parameters': {
                'num_captions': args.num_captions,
                'method': args.method,
                'max_length': args.max_length,
                'beam_size': args.beam_size,
                'temperature': args.temperature
            }
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Results saved to: {args.output}")


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="Generate video captions")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--device", type=str, help="Device to use (cuda/cpu)")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    
    # Generation parameters
    parser.add_argument("--method", type=str, default="greedy", choices=["greedy", "beam"], 
                       help="Generation method")
    parser.add_argument("--max-length", type=int, default=20, help="Maximum caption length")
    parser.add_argument("--beam-size", type=int, default=5, help="Beam size for beam search")
    parser.add_argument("--length-penalty", type=float, default=1.0, help="Length penalty for beam search")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling")
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Single video prediction
    single_parser = subparsers.add_parser("single", help="Predict caption for single video")
    single_parser.add_argument("--video-path", type=str, required=True, help="Path to video file")
    single_parser.add_argument("--features-path", type=str, help="Path to pre-extracted features")
    single_parser.add_argument("--output", type=str, help="Output JSON file")
    
    # Batch prediction
    batch_parser = subparsers.add_parser("batch", help="Predict captions for multiple videos")
    batch_parser.add_argument("--video-list", type=str, required=True, 
                             help="Path to text file with video paths or directory with videos")
    batch_parser.add_argument("--batch-size", type=int, default=8, help="Batch size for processing")
    batch_parser.add_argument("--output", type=str, help="Output JSON file")
    batch_parser.add_argument("--captions-file", type=str, help="Output text file with captions only")
    
    # Multiple captions for single video
    multiple_parser = subparsers.add_parser("multiple", help="Generate multiple captions for single video")
    multiple_parser.add_argument("--video-path", type=str, required=True, help="Path to video file")
    multiple_parser.add_argument("--features-path", type=str, help="Path to pre-extracted features")
    multiple_parser.add_argument("--num-captions", type=int, default=5, help="Number of captions to generate")
    multiple_parser.add_argument("--output", type=str, help="Output JSON file")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Execute command
    try:
        if args.command == "single":
            predict_single_video(args)
        elif args.command == "batch":
            predict_batch_videos(args)
        elif args.command == "multiple":
            predict_multiple_captions(args)
        
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise


if __name__ == "__main__":
    main()