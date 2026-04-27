#!/usr/bin/env python3
"""
Fisheye Camera Calibration from Images
Loads images from 'cam' folder and performs fisheye calibration
Uses OpenCV fisheye model for wide-angle/fisheye lenses
"""

import numpy as np
import cv2 as cv
import glob
import json
import os
from datetime import datetime

def fisheye_calibrate_from_images(image_folder='cam', chessboard_size=(7, 6), square_size=1.0, output_file='fisheye_calibration.json'):
    """
    Calibrate fisheye camera from images in a folder
    
    Args:
        image_folder: Folder containing calibration images
        chessboard_size: Number of inner corners (width, height)
        square_size: Size of chessboard square (in your units, e.g., mm)
        output_file: Output file for calibration data
    """
    
    print("=" * 60)
    print("Fisheye Camera Calibration from Images")
    print("=" * 60)
    print(f"\nImage folder: {image_folder}/")
    print(f"Chessboard size: {chessboard_size[0]}x{chessboard_size[1]} inner corners")
    print(f"Square size: {square_size}")
    print(f"Using: OpenCV Fisheye Model")
    
    # Check if folder exists
    if not os.path.exists(image_folder):
        print(f"\nError: Folder '{image_folder}' not found!")
        print(f"Please create the folder and add calibration images.")
        return False
    
    # Termination criteria for corner refinement
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size
    
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane
    
    # Load images
    image_patterns = [
        os.path.join(image_folder, '*.jpg'),
        os.path.join(image_folder, '*.jpeg'),
        os.path.join(image_folder, '*.png'),
        os.path.join(image_folder, '*.bmp')
    ]
    
    images = []
    for pattern in image_patterns:
        images.extend(glob.glob(pattern))
    
    if not images:
        print(f"\nError: No images found in '{image_folder}/'")
        print("Supported formats: .jpg, .jpeg, .png, .bmp")
        return False
    
    print(f"\nFound {len(images)} images")
    print("\nProcessing images...\n")
    
    successful_images = []
    failed_images = []
    
    for i, fname in enumerate(images, 1):
        img = cv.imread(fname)
        if img is None:
            print(f"[{i}/{len(images)}] ✗ {os.path.basename(fname)} - Cannot read image")
            failed_images.append(fname)
            continue
        
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, chessboard_size, 
                                               cv.CALIB_CB_ADAPTIVE_THRESH + 
                                               cv.CALIB_CB_FAST_CHECK + 
                                               cv.CALIB_CB_NORMALIZE_IMAGE)
        
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            
            # Refine corner positions
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            
            print(f"[{i}/{len(images)}] ✓ {os.path.basename(fname)} - Pattern detected")
            successful_images.append(fname)
            
            # Draw and display the corners (optional)
            cv.drawChessboardCorners(img, chessboard_size, corners2, ret)
            cv.imshow('Detected Corners', img)
            cv.waitKey(100)  # Show for 100ms
        else:
            print(f"[{i}/{len(images)}] ✗ {os.path.basename(fname)} - Pattern not found")
            failed_images.append(fname)
        if len(successful_images) > 100:
            break
    cv.destroyAllWindows()
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Successful: {len(successful_images)}/{len(images)}")
    print(f"Failed: {len(failed_images)}/{len(images)}")
    
    if failed_images:
        print(f"\nFailed images:")
        for fname in failed_images[:10]:  # Show first 10
            print(f"  - {os.path.basename(fname)}")
        if len(failed_images) > 10:
            print(f"  ... and {len(failed_images) - 10} more")
    
    if len(successful_images) < 10:
        print(f"\n✗ Error: Not enough valid images ({len(successful_images)})")
        print("  Need at least 10 images with detected chessboard pattern")
        print("\nTips for fisheye calibration:")
        print("  - Use MORE images than standard calibration (30-50+ recommended)")
        print("  - Cover entire field of view, especially edges and corners")
        print("  - Include images at various distances")
        print("  - Ensure chessboard fills different amounts of frame (20-80%)")
        return False
    
    print(f"\n{'='*60}")
    print(f"Performing FISHEYE calibration with {len(successful_images)} images...")
    print(f"{'='*60}\n")
    
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 1, 3), np.float32)
    objp[:, 0, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size
    
    # Rebuild objpoints list (each image gets identical objp)
    objpoints = [objp for _ in range(len(imgpoints))]
    
    # 2. Image points: convert list of (N,1,2) or (N,2) → list of (N,1,2)
    imgpoints_fixed = []
    for corners in imgpoints:
        if corners.ndim == 2:  # (N,2) → (N,1,2)
            corners = corners.reshape(-1, 1, 2)
        elif corners.shape[1] == 2:  # Ensure (N,1,2)
            corners = corners.reshape(-1, 1, 2)
        imgpoints_fixed.append(corners.astype(np.float32))
    
    # Debug shapes (remove after fixing)
    print("DEBUG shapes:")
    print(f"objpoints[0].shape: {objpoints[0].shape}, dtype: {objpoints[0].dtype}")
    print(f"imgpoints_fixed[0].shape: {imgpoints_fixed[0].shape}, dtype: {imgpoints_fixed[0].dtype}")
    
    # 3. Initialize outputs
    h, w = gray.shape
    # K = np.zeros((3, 3))
    K = np.array([
            [max(w, h) * 0.8,  0.0,              w / 2.0],
            [0.0,              max(w, h) * 0.8,  h / 2.0],
            [0.0,              0.0,              1.0     ],
        ], dtype=np.float64)
    # K[0,0] = K[1,1] = max(w,h) * 0.8  # fx = fy = 80% of image diagonal
    # K[0,2] = w * 0.5                   # cx = image center X  
    # K[1,2] = h * 0.5                   # cy = image center Y

    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), np.float32) for _ in imgpoints_fixed]
    tvecs = [np.zeros((1, 1, 3), np.float32) for _ in imgpoints_fixed]
    
    flags = cv.fisheye.CALIB_RECOMPUTE_EXTRINSIC 
    
    # 4. Calibrate
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv.fisheye.calibrate(
        objpoints,           # list[N] of (42,1,3) float32
        imgpoints_fixed,     # list[N] of (42,1,2) float32  
        (w, h),
        K, D, rvecs, tvecs,
        flags,
        (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )
    print("✓ Fisheye calibration successful!\n")
    print(f"RMS Reprojection Error: {rms:.4f} pixels")
    
    if rms < 0.5:
        print("  → Excellent fisheye calibration!")
    elif rms < 1.0:
        print("  → Good fisheye calibration")
    elif rms < 2.0:
        print("  → Acceptable fisheye calibration")
    else:
        print("  → Consider recalibrating with more/better images")
    
    print(f"\nCamera Matrix (K):")
    print(camera_matrix)
    print(f"\nDistortion Coefficients (D) - Fisheye Model [k1, k2, k3, k4]:")
    print(dist_coeffs.ravel())
    
    # Save calibration data
    calibration_data = {
        'calibration_model': 'fisheye',
        'camera_matrix': camera_matrix.tolist(),
        'dist_coeffs': dist_coeffs.tolist(),
        'rms_error': float(rms),
        'image_size': [int(gray.shape[1]), int(gray.shape[0])],  # [width, height]
        'num_images': len(successful_images),
        'chessboard_size': list(chessboard_size),
        'square_size': float(square_size),
        'calibration_date': datetime.now().isoformat(),
        'successful_images': [os.path.basename(f) for f in successful_images],
        'opencv_model': 'cv2.fisheye'
    }
    
    with open(output_file, 'w') as f:
        json.dump(calibration_data, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✓ Fisheye calibration data saved to: {output_file}")
    print(f"{'='*60}\n")
    
    return True


def test_fisheye_calibration(image_path, calibration_file='fisheye_calibration.json'):
    """
    Test fisheye calibration by undistorting an image
    
    Args:
        image_path: Path to test image
        calibration_file: Path to calibration JSON file
    """
    # Load calibration data
    try:
        with open(calibration_file, 'r') as f:
            calib_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Calibration file not found: {calibration_file}")
        return
    
    if calib_data.get('calibration_model') != 'fisheye':
        print("Warning: This calibration is not a fisheye model!")
        print("Use the standard calibration test instead.")
        return
    
    K = np.array(calib_data['camera_matrix'])
    D = np.array(calib_data['dist_coeffs'])
    
    # Load image
    img = cv.imread(image_path)
    if img is None:
        print(f"Error: Cannot read image {image_path}")
        return
    
    h, w = img.shape[:2]
    
    print(f"Undistorting fisheye image: {image_path}")
    print(f"Image size: {w}x{h}")
    
    # Compute new camera matrix for undistortion
    # balance=0: retain all pixels, balance=1: crop to valid area
    new_K = cv.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (w, h), np.eye(3), balance=1
    )
    
    # Create undistortion maps
    map1, map2 = cv.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, (w, h), cv.CV_16SC2
    )
    
    # Undistort image
    undistorted = cv.remap(img, map1, map2, interpolation=cv.INTER_LINEAR, 
                           borderMode=cv.BORDER_CONSTANT)
    
    # Display comparison
    # Resize if image is too large
    max_width = 1920
    if w > max_width:
        scale = max_width / w
        new_size = (int(w * scale), int(h * scale))
        img_display = cv.resize(img, new_size)
        undistorted_display = cv.resize(undistorted, new_size)
    else:
        img_display = img
        undistorted_display = undistorted
    
    comparison = np.hstack((img_display, undistorted_display))
    cv.putText(comparison, "Original (Fisheye)", (10, 30), 
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv.putText(comparison, "Undistorted", (img_display.shape[1] + 10, 30), 
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv.imshow('Fisheye Calibration Test - Press any key to close', comparison)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    # Save result
    output_path = 'fisheye_undistorted_test.jpg'
    cv.imwrite(output_path, undistorted)
    print(f"✓ Undistorted image saved to: {output_path}")


def compare_standard_vs_fisheye(image_path, standard_calib='camera_calibration.json', 
                                fisheye_calib='fisheye_calibration.json'):
    """
    Compare standard and fisheye calibration models side by side
    """
    img = cv.imread(image_path)
    if img is None:
        print(f"Error: Cannot read image {image_path}")
        return
    
    h, w = img.shape[:2]
    
    # Load standard calibration
    try:
        with open(standard_calib, 'r') as f:
            std_calib = json.load(f)
        
        std_K = np.array(std_calib['camera_matrix'])
        std_D = np.array(std_calib['dist_coeffs'])
        
        # Undistort with standard model
        new_std_K, roi = cv.getOptimalNewCameraMatrix(std_K, std_D, (w, h), 1, (w, h))
        std_undistorted = cv.undistort(img, std_K, std_D, None, new_std_K)
    except:
        std_undistorted = None
        print("Could not load standard calibration")
    
    # Load fisheye calibration
    try:
        with open(fisheye_calib, 'r') as f:
            fish_calib = json.load(f)
        
        fish_K = np.array(fish_calib['camera_matrix'])
        fish_D = np.array(fish_calib['dist_coeffs'])
        
        # Undistort with fisheye model
        new_fish_K = cv.fisheye.estimateNewCameraMatrixForUndistortRectify(
            fish_K, fish_D, (w, h), np.eye(3), balance=0
        )
        map1, map2 = cv.fisheye.initUndistortRectifyMap(
            fish_K, fish_D, np.eye(3), new_fish_K, (w, h), cv.CV_16SC2
        )
        fish_undistorted = cv.remap(img, map1, map2, interpolation=cv.INTER_LINEAR)
    except:
        fish_undistorted = None
        print("Could not load fisheye calibration")
    
    # Create comparison
    if std_undistorted is not None and fish_undistorted is not None:
        comparison = np.hstack((img, std_undistorted, fish_undistorted))
        cv.putText(comparison, "Original", (10, 30), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv.putText(comparison, "Standard Model", (w + 10, 30), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv.putText(comparison, "Fisheye Model", (2*w + 10, 30), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    elif fish_undistorted is not None:
        comparison = np.hstack((img, fish_undistorted))
        cv.putText(comparison, "Original", (10, 30), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv.putText(comparison, "Fisheye Model", (w + 10, 30), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        print("No calibration data available for comparison")
        return
    
    cv.imshow('Model Comparison - Press any key to close', comparison)
    cv.waitKey(0)
    cv.destroyAllWindows()


def main():
    """Main function"""
    print("\n" + "=" * 60)
    print("Fisheye Camera Calibration Tool")
    print("=" * 60)
    
    while True:
        print("\nOptions:")
        print("1. Fisheye calibration from 'cam' folder")
        print("2. Fisheye calibration from custom folder")
        print("3. Test fisheye calibration on an image")
        print("4. Compare standard vs fisheye calibration")
        print("5. Exit")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == '1':
            # Default fisheye calibration from 'cam' folder
            fisheye_calibrate_from_images(image_folder='cam', chessboard_size=(8, 6))
        
        elif choice == '2':
            # Custom folder
            folder = input("Enter folder path: ").strip()
            width = input("Chessboard width (inner corners, default 7): ").strip()
            height = input("Chessboard height (inner corners, default 6): ").strip()
            
            width = int(width) if width else 7
            height = int(height) if height else 6
            
            fisheye_calibrate_from_images(image_folder=folder, chessboard_size=(width, height))
        
        elif choice == '3':
            # Test fisheye calibration
            if not os.path.exists('fisheye_calibration.json'):
                print("\nError: No fisheye calibration file found!")
                print("Please calibrate first (option 1 or 2)")
                continue
            
            image_path = input("Enter test image path: ").strip()
            if os.path.exists(image_path):
                test_fisheye_calibration(image_path)
            else:
                print(f"Error: Image not found: {image_path}")
        
        elif choice == '4':
            # Compare models
            image_path = input("Enter test image path: ").strip()
            if os.path.exists(image_path):
                compare_standard_vs_fisheye(image_path)
            else:
                print(f"Error: Image not found: {image_path}")
        
        elif choice == '5':
            print("\nExiting...")
            break
        
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()