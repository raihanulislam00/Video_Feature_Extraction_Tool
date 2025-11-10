# Example Usage Script
# This script demonstrates how to use the Video Feature Extractor

from video_feature_extractor import VideoFeatureExtractor
import sys

def main():
    """
    Example script showing different ways to use the video feature extractor.
    """
    
    # Example 1: Quick analysis of a single video
    print("="*60)
    print("Example 1: Complete Feature Extraction")
    print("="*60)
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = "sample_video.mp4"  # Default video path
        print(f"No video provided. Using default: {video_path}")
        print("Usage: python example_usage.py <path_to_video>")
        return
    
    # Initialize extractor with sampling every 30 frames
    extractor = VideoFeatureExtractor(video_path, sample_rate=30)
    
    # Extract all features
    features = extractor.extract_all_features()
    
    # Display summary
    extractor.print_summary()
    
    # Save results
    output_file = extractor.save_features()
    
    # Example 2: Individual feature extraction with custom parameters
    print("\n" + "="*60)
    print("Example 2: Custom Feature Extraction")
    print("="*60)
    
    # Reinitialize for fresh extraction
    extractor2 = VideoFeatureExtractor(video_path, sample_rate=15)
    
    # Extract shot cuts with custom threshold
    print("\n1. Detecting shot cuts with higher sensitivity...")
    cuts = extractor2.detect_shot_cuts(threshold=25.0)
    print(f"   Found {cuts['total_cuts']} cuts")
    
    # Extract motion with more samples
    print("\n2. Analyzing motion...")
    motion = extractor2.analyze_motion()
    print(f"   Motion level: {motion['motion_category']}")
    
    # Extract text from more frames
    print("\n3. Detecting text...")
    text = extractor2.detect_text(sample_frames=30)
    if 'error' not in text:
        print(f"   Text found in {text['text_present_ratio']:.1%} of frames")
    
    # Detect objects with higher confidence
    print("\n4. Detecting objects and people...")
    objects = extractor2.detect_objects_and_people(confidence=0.6, sample_frames=40)
    if 'error' not in objects:
        print(f"   Dominance: {objects['dominance']}")
    
    print("\n" + "="*60)
    print("Examples completed successfully!")
    print("="*60)
    

if __name__ == "__main__":
    main()
