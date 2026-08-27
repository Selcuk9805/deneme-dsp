import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.services.downloader import DownloaderService
from app.services.analyzer import AnalyzerService
from app.services.transition import TransitionService

def main():
    parser = argparse.ArgumentParser(description="Test Automix DSP Backend")
    parser.add_argument("track_a", help="URL or path for Track A")
    parser.add_argument("track_b", help="URL or path for Track B")
    
    args = parser.parse_args()
    
    print(f"Downloading/Locating Track A: {args.track_a}")
    path_a = DownloaderService.get_audio_path(args.track_a)
    print(f"Track A located at: {path_a}")
    
    print(f"Downloading/Locating Track B: {args.track_b}")
    path_b = DownloaderService.get_audio_path(args.track_b)
    print(f"Track B located at: {path_b}")
    
    print("Analyzing Track A...")
    features_a = AnalyzerService.analyze_track(path_a, is_track_a=True)
    
    print("Analyzing Track B...")
    features_b = AnalyzerService.analyze_track(path_b, is_track_a=False)
    
    print("Calculating Transition...")
    plan = TransitionService.calculate_transition(features_a, features_b)
    
    print("\n=== TRANSITION PLAN ===")
    print(plan.model_dump_json(indent=2))

if __name__ == "__main__":
    main()
