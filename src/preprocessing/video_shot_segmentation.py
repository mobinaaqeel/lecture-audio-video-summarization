"""
video_shot_segmentation.py
-------------------------------------------------
Step 1: Video Shot Segmentation using YCbCr Color Model
-------------------------------------------------
How it works:
  1. Convert each frame from BGR → YCbCr color space
  2. Calculate 3 color moments per channel (mean, std, skewness)
     → 3 channels × 3 moments = 9 features per frame
  3. Calculate dissimilarity between consecutive frames
  4. If dissimilarity > threshold → Shot boundary detected
  5. Group frames into shots (segments)
-------------------------------------------------
Reference: Paper Section 3.2
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import json
import time


# ============================================================
# CLASS 1: Color Moment Calculator
# ============================================================
class ColorMomentCalculator:
    """
    Calculate color moments in YCbCr space.
    
    For each frame we compute 9 values:
        Y channel  → mean, std, skewness
        Cb channel → mean, std, skewness
        Cr channel → mean, std, skewness
    
    Paper equations:
        Ed  = mean            (Eq. 2)
        Cd  = std deviation   (Eq. 3)
        Bd  = skewness        (Eq. 4)
    """
    
    def __init__(self):
        pass
    
    def bgr_to_ycbcr(self, frame: np.ndarray) -> np.ndarray:
        """
        Convert BGR frame to YCbCr color space.
        
        Args:
            frame: BGR image (H, W, 3)
        Returns:
            YCbCr image (H, W, 3)
        """
        return cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    
    def compute_mean(self, channel: np.ndarray) -> float:
        """
        Paper Eq. 2: Ed = (1/D) * sum(Gd,e)
        Mean of pixel values in a channel.
        """
        return np.mean(channel).astype(float)
    
    def compute_std(self, channel: np.ndarray) -> float:
        """
        Paper Eq. 3: Cd = sqrt( (1/D) * sum((Gd,e - Ed)^2) )
        Standard deviation of pixel values.
        """
        return np.std(channel).astype(float)
    
    def compute_skewness(self, channel: np.ndarray) -> float:
        """
        Paper Eq. 4: Bd = cbrt( (1/D) * sum((Gd,e - Ed)^3) )
        Skewness measures asymmetry of the distribution.
        """
        mean = np.mean(channel)
        diff = (channel.astype(float) - mean) ** 3
        skew = np.cbrt(np.mean(diff))
        return float(skew)
    
    def get_color_moments(self, frame: np.ndarray) -> np.ndarray:
        """
        Get all 9 color moments for a frame.
        
        Args:
            frame: BGR image
        Returns:
            Array of 9 values [Y_mean, Y_std, Y_skew,
                               Cb_mean, Cb_std, Cb_skew,
                               Cr_mean, Cr_std, Cr_skew]
        """
        # Convert to YCbCr
        ycbcr = self.bgr_to_ycbcr(frame)
        
        moments = []
        
        # For each channel (Y, Cb, Cr)
        for ch in range(3):
            channel = ycbcr[:, :, ch]
            moments.append(self.compute_mean(channel))
            moments.append(self.compute_std(channel))
            moments.append(self.compute_skewness(channel))
        
        return np.array(moments)


# ============================================================
# CLASS 2: Dissimilarity Calculator
# ============================================================
class DissimilarityCalculator:
    """
    Calculate dissimilarity between two frames
    using weighted difference of color moments.
    
    Paper Eq. 5: weighted sum of moment differences
    """
    
    def __init__(self, weights: List[float] = None):
        """
        Args:
            weights: 9 weights for each moment.
                     Default: equal weights
        """
        if weights is None:
            # Default equal weights
            self.weights = np.ones(9) / 9.0
        else:
            self.weights = np.array(weights)
    
    def compute_dissimilarity(self, moments1: np.ndarray, moments2: np.ndarray) -> float:
        """
        Compute weighted dissimilarity between two moment vectors.
        
        Args:
            moments1: Color moments of frame 1 (9 values)
            moments2: Color moments of frame 2 (9 values)
        Returns:
            Dissimilarity score (float)
        """
        diff = np.abs(moments1 - moments2)
        dissimilarity = np.sum(self.weights * diff)
        return float(dissimilarity)


# ============================================================
# CLASS 3: Shot Segmenter (Main Class)
# ============================================================
class VideoShotSegmenter:
    """
    Main class for video shot segmentation.
    
    Pipeline:
        Input Video → Extract Frames → Compute Moments
        → Calculate Dissimilarity → Detect Boundaries → Output Shots
    """
    
    def __init__(self, threshold: float = 30.0, skip_frames: int = 1):
        """
        Args:
            threshold: Dissimilarity threshold for shot boundary.
                       Higher = fewer shots, Lower = more shots
            skip_frames: Process every Nth frame (1 = every frame)
        """
        self.threshold = threshold
        self.skip_frames = skip_frames
        self.color_calc = ColorMomentCalculator()
        self.dissim_calc = DissimilarityCalculator()
    
    def load_video(self, video_path: str) -> cv2.VideoCapture:
        """Load video file"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        return cap
    
    def extract_all_moments(self, video_path: str) -> Tuple[List[np.ndarray], List[int]]:
        """
        Extract color moments for all frames.
        
        Args:
            video_path: Path to video file
        Returns:
            moments: List of moment arrays (one per frame)
            frame_indices: List of frame numbers processed
        """
        cap = self.load_video(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        moments = []
        frame_indices = []
        frame_num = 0
        
        print(f"  📊 Extracting color moments from {total_frames} frames...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames if needed (for speed)
            if frame_num % self.skip_frames == 0:
                m = self.color_calc.get_color_moments(frame)
                moments.append(m)
                frame_indices.append(frame_num)
            
            frame_num += 1
            
            # Progress update every 500 frames
            if frame_num % 500 == 0:
                print(f"     Processed {frame_num}/{total_frames} frames...")
        
        cap.release()
        print(f"  ✅ Extracted moments for {len(moments)} frames")
        
        return moments, frame_indices
    
    def detect_boundaries(self, moments: List[np.ndarray]) -> List[int]:
        """
        Detect shot boundaries using dissimilarity threshold.
        
        Args:
            moments: List of color moment arrays
        Returns:
            List of boundary frame indices
        """
        boundaries = [0]  # First frame is always a boundary
        
        for i in range(1, len(moments)):
            dissim = self.dissim_calc.compute_dissimilarity(moments[i-1], moments[i])
            
            if dissim > self.threshold:
                boundaries.append(i)
        
        boundaries.append(len(moments) - 1)  # Last frame is boundary
        
        return boundaries
    
    def get_shots(self, boundaries: List[int], frame_indices: List[int]) -> List[Dict]:
        """
        Create shot segments from boundaries.
        
        Args:
            boundaries: List of boundary positions in moments list
            frame_indices: Actual frame numbers
        Returns:
            List of shot dictionaries
        """
        shots = []
        
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]
            
            shot = {
                "shot_id": i,
                "start_frame": frame_indices[start_idx],
                "end_frame": frame_indices[end_idx],
                "num_frames": end_idx - start_idx,
                "duration_frames": frame_indices[end_idx] - frame_indices[start_idx]
            }
            shots.append(shot)
        
        return shots
    
    def segment_video(self, video_path: str) -> Tuple[List[Dict], List[np.ndarray]]:
        """
        Main function: Segment a video into shots.
        
        Args:
            video_path: Path to video file
        Returns:
            shots: List of shot dictionaries
            moments: All computed color moments
        """
        print(f"\n🎬 Segmenting: {Path(video_path).name}")
        print("-" * 50)
        
        # Step 1: Extract moments
        moments, frame_indices = self.extract_all_moments(video_path)
        
        # Step 2: Detect boundaries
        print("  🔍 Detecting shot boundaries...")
        boundaries = self.detect_boundaries(moments)
        print(f"  ✅ Found {len(boundaries) - 1} shot boundaries")
        
        # Step 3: Create shots
        shots = self.get_shots(boundaries, frame_indices)
        
        # Print results
        print(f"\n  📊 SEGMENTATION RESULTS:")
        print(f"  {'─' * 45}")
        print(f"  {'Shot':<8} {'Start':<10} {'End':<10} {'Frames':<10}")
        print(f"  {'─' * 45}")
        for shot in shots:
            print(f"  {shot['shot_id']:<8} {shot['start_frame']:<10} "
                  f"{shot['end_frame']:<10} {shot['num_frames']:<10}")
        print(f"  {'─' * 45}")
        print(f"  Total Shots: {len(shots)}")
        
        return shots, moments
    
    def save_results(self, shots: List[Dict], video_path: str, output_dir: str = "data/processed"):
        """Save segmentation results to JSON"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        video_name = Path(video_path).stem
        result = {
            "video": video_name,
            "threshold": self.threshold,
            "total_shots": len(shots),
            "shots": shots
        }
        
        save_path = output_path / f"{video_name}_shots.json"
        with open(save_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\n  💾 Results saved: {save_path}")
    
    def extract_shot_frames(self, video_path: str, shots: List[Dict],
                            output_dir: str = "outputs/frames"):
        """
        Extract one representative frame per shot.
        
        Args:
            video_path: Path to video
            shots: List of shot dictionaries
            output_dir: Where to save frames
        """
        output_path = Path(output_dir) / Path(video_path).stem
        output_path.mkdir(parents=True, exist_ok=True)
        
        cap = self.load_video(video_path)
        
        print(f"\n  📸 Extracting representative frames...")
        
        for shot in shots:
            # Get middle frame of each shot
            mid_frame = (shot['start_frame'] + shot['end_frame']) // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
            ret, frame = cap.read()
            
            if ret:
                frame_path = output_path / f"shot_{shot['shot_id']:03d}_frame_{mid_frame:06d}.jpg"
                cv2.imwrite(str(frame_path), frame)
        
        cap.release()
        print(f"  ✅ Saved {len(shots)} frames to {output_path}")


# ============================================================
# FUNCTION: Visualize Shots (for Colab)
# ============================================================
def visualize_shots(video_path: str, shots: List[Dict], max_shots: int = 10):
    """
    Display representative frame from each shot in Colab.
    
    Args:
        video_path: Path to video
        shots: List of shot dicts
        max_shots: Maximum shots to display
    """
    import matplotlib.pyplot as plt
    from google.colab.patches import cv2_imshow
    
    cap = cv2.VideoCapture(video_path)
    
    n = min(len(shots), max_shots)
    cols = 3
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4))
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes.flatten()
    
    for i in range(n):
        shot = shots[i]
        mid_frame = (shot['start_frame'] + shot['end_frame']) // 2
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
        ret, frame = cap.read()
        
        if ret:
            # BGR to RGB for matplotlib
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            axes[i].imshow(frame_rgb)
            axes[i].set_title(f"Shot {shot['shot_id']}\n"
                              f"Frames: {shot['start_frame']}-{shot['end_frame']}")
            axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(n, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle("📹 Video Shot Segmentation Results", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig("outputs/results/shot_segmentation.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    cap.release()
    print("✅ Visualization saved to outputs/results/shot_segmentation.png")


def plot_dissimilarity(video_path: str, moments: List[np.ndarray], threshold: float):
    """
    Plot frame-to-frame dissimilarity to visualize boundaries.
    
    Args:
        video_path: Video name for title
        moments: Color moments list
        threshold: Threshold used
    """
    import matplotlib.pyplot as plt
    
    dissim_calc = DissimilarityCalculator()
    
    dissimilarities = []
    for i in range(1, len(moments)):
        d = dissim_calc.compute_dissimilarity(moments[i-1], moments[i])
        dissimilarities.append(d)
    
    plt.figure(figsize=(15, 4))
    plt.plot(dissimilarities, color='steelblue', linewidth=0.8, label='Dissimilarity')
    plt.axhline(y=threshold, color='red', linestyle='--', linewidth=1.5, label=f'Threshold ({threshold})')
    
    # Mark boundaries
    for i, d in enumerate(dissimilarities):
        if d > threshold:
            plt.axvline(x=i, color='green', alpha=0.3, linewidth=1)
    
    plt.xlabel("Frame Number", fontsize=12)
    plt.ylabel("Dissimilarity Score", fontsize=12)
    plt.title(f"📊 Frame Dissimilarity - {Path(video_path).name}", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/results/dissimilarity_plot.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ Plot saved to outputs/results/dissimilarity_plot.png")


# ============================================================
# MAIN: Run in Colab
# ============================================================
def run_segmentation(video_path: str, threshold: float = 30.0, skip_frames: int = 1):
    """
    Main entry point - run full segmentation pipeline.
    
    Args:
        video_path: Path to your video file
        threshold: Shot boundary threshold (tune this!)
        skip_frames: Process every Nth frame
    """
    import os
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    print("=" * 55)
    print("🎬 VIDEO SHOT SEGMENTATION (YCbCr Color Model)")
    print("=" * 55)
    print(f"  Video    : {Path(video_path).name}")
    print(f"  Threshold: {threshold}")
    print(f"  Skip     : Every {skip_frames} frame(s)")
    print("=" * 55)
    
    # Create segmenter
    segmenter = VideoShotSegmenter(threshold=threshold, skip_frames=skip_frames)
    
    # Run segmentation
    shots, moments = segmenter.segment_video(video_path)
    
    # Save results
    segmenter.save_results(shots, video_path)
    
    # Extract frames
    segmenter.extract_shot_frames(video_path, shots)
    
    # Visualize
    print("\n📊 Generating visualizations...")
    plot_dissimilarity(video_path, moments, threshold)
    visualize_shots(video_path, shots)
    
    print("\n" + "=" * 55)
    print("✅ SEGMENTATION COMPLETE!")
    print("=" * 55)
    
    return shots, moments