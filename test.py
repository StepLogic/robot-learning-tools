# === FISHEYE MODEL (for comparison) ===
fish_K = np.array([[191.8682, 0, 320], [0, 191.8682, 240], [0, 0, 1]], dtype=np.float32)
fish_D = np.array([[-0.0018], [0], [0], [0]], dtype=np.float32)
new_fish_K = cv.fisheye.estimateNewCameraMatrixForUndistortRectify(fish_K, fish_D, (w, h), np.eye(3), 0.75)
map1, map2 = cv.fisheye.initUndistortRectifyMap(fish_K, fish_D, np.eye(3), new_fish_K, (w, h), cv.CV_16SC2)
undistorted_fish = cv.remap(img, map1, map2, cv.INTER_LINEAR)
