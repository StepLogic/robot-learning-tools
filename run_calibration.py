#!/usr/bin/env python3
"""
Simple Camera Stream and Frame Capture
Automatically saves all frames from the camera
"""

import cv2
import numpy as np
import os
from datetime import datetime

class CameraStreamer:
    def __init__(self, camera_device='/dev/video2'):
        """
        Initialize camera streamer
        
        Args:
            camera_device: Path to camera device
        """
        self.camera_device = camera_device
        
    def stream_and_save(self, output_dir='captured_frames', save_interval=1):
        """
        Stream video and save all frames
        
        Args:
            output_dir: Directory to save frames
            save_interval: Save every Nth frame (1 = save all frames)
        """
        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")
        
        cap = cv2.VideoCapture(self.camera_device)
        if not cap.isOpened():
            print(f"Error: Cannot open camera at {self.camera_device}")
            return
        
        # Get camera properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"\nCamera: {self.camera_device}")
        print(f"Resolution: {width}x{height}")
        print(f"FPS: {fps}")
        print(f"Saving to: {output_dir}/")
        print(f"Save interval: every {save_interval} frame(s)")
        print("\nControls:")
        print("  'q' - Quit")
        print("  's' - Manual snapshot")
        print("  SPACE - Pause/Resume auto-saving")
        print("\nStreaming...\n")
        
        frame_count = 0
        saved_count = 0
        manual_snapshot_count = 0
        auto_save_enabled = True
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame")
                break
            
            frame_count += 1
            
            # Auto-save frames at specified interval
            if auto_save_enabled and frame_count % save_interval == 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = os.path.join(output_dir, f"frame_{saved_count:06d}_{timestamp}.jpg")
                cv2.imwrite(filename, frame)
                saved_count += 1
            
            # Add info overlay
            display_frame = frame.copy()
            status = "SAVING" if auto_save_enabled else "PAUSED"
            color = (0, 255, 0) if auto_save_enabled else (0, 165, 255)
            
            cv2.putText(display_frame, f"Status: {status}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(display_frame, f"Frames: {frame_count}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Saved: {saved_count}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, "Press 'q' to quit", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            cv2.imshow('Camera Stream', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Manual snapshot
                manual_snapshot_count += 1
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = os.path.join(output_dir, f"snapshot_{manual_snapshot_count:03d}_{timestamp}.jpg")
                cv2.imwrite(filename, frame)
                print(f"Manual snapshot saved: {filename}")
            elif key == 32:  # SPACE
                # Toggle auto-save
                auto_save_enabled = not auto_save_enabled
                status_text = "enabled" if auto_save_enabled else "paused"
                print(f"Auto-save {status_text}")
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n{'='*60}")
        print("Stream ended")
        print(f"Total frames processed: {frame_count}")
        print(f"Auto-saved frames: {saved_count}")
        print(f"Manual snapshots: {manual_snapshot_count}")
        print(f"Output directory: {output_dir}/")
        print(f"{'='*60}")


def main():
    """Main function"""
    print("=" * 60)
    print("Simple Camera Stream and Frame Capture")
    print("=" * 60)
    
    # Get camera device
    camera_device = input("\nCamera device (default: /dev/video2): ").strip()
    camera_device = camera_device if camera_device else '/dev/video2'
    
    # Get output directory
    output_dir = input("Output directory (default: captured_frames): ").strip()
    output_dir = output_dir if output_dir else 'captured_frames'
    
    # Get save interval
    save_interval_input = input("Save interval - every Nth frame (default: 1 = all frames): ").strip()
    try:
        save_interval = int(save_interval_input) if save_interval_input else 1
        if save_interval < 1:
            save_interval = 1
    except ValueError:
        save_interval = 1
        print("Invalid input, using default: 1")
    
    # Initialize and start streaming
    streamer = CameraStreamer(camera_device=camera_device)
    streamer.stream_and_save(output_dir=output_dir, save_interval=save_interval)


if __name__ == "__main__":
    main()