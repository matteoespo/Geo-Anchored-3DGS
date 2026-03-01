import cv2
import numpy as np

class VisualOdometry:
    def __init__(self, camera_intrinsics: np.ndarray):
        """
        Initializes the Visual Odometry pipeline.
        :param camera_intrinsics: 3x3 Camera Matrix (K) containing focal lengths and optical centers.
        """
        self.K = camera_intrinsics
        
        # Initialize ORB detector. We keep up to 3000 features for good accuracy.
        self.orb = cv2.ORB_create(nfeatures=3000)
        
        # Initialize Brute-Force Matcher with Hamming distance (best for ORB binary descriptors)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def extract_features(self, image: np.ndarray):
        """Finds keypoints and their descriptors in an image."""
        # Convert to grayscale if it's a color image
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        return keypoints, descriptors

    def match_features(self, desc1, desc2):
        """Matches descriptors between two frames using Lowe's ratio test."""
        # k=2 means we find the top 2 best matches for each descriptor
        raw_matches = self.matcher.knnMatch(desc1, desc2, k=2)
        
        good_matches = []
        # Lowe's ratio test: a match is reliable if the best match is significantly 
        # better than the second best match.
        for m, n in raw_matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
                
        return good_matches

    def estimate_motion(self, kp1, kp2, matches):
        """
        Calculates Rotation (R) and Translation (t) between two frames.
        """
        # Extract the (x, y) coordinates of the matching keypoints
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        # Find the Essential Matrix using RANSAC (to ignore bad matches/outliers)
        E, mask = cv2.findEssentialMat(pts2, pts1, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        
        # Recover Rotation and Translation from the Essential Matrix
        _, R, t, mask = cv2.recoverPose(E, pts2, pts1, self.K)
        
        return R, t, pts1, pts2, mask
    
    def triangulate_points(self, R, t, pts1, pts2):
        """
        Triangulates 2D matched points into 3D space.
        
        :param R: Rotation matrix from frame 1 to 2
        :param t: Translation vector from frame 1 to 2
        :param pts1: 2D points in frame 1
        :param pts2: 2D points in frame 2
        :return: 3D points (N x 3 array) in the local coordinate system of frame 1
        """
        # P1: Projection matrix for the first camera (Identity rotation, zero translation)
        # K is 3x3, [I | 0] is 3x4 -> P1 is 3x4
        P1 = self.K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        
        # P2: Projection matrix for the second camera (Moved by R, t)
        P2 = self.K @ np.hstack((R, t))
        
        # Triangulate points (returns homogeneous coordinates 4D: X, Y, Z, W)
        pts4D = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        
        # Convert from homogeneous to 3D Euclidean space by dividing by W
        pts3D = pts4D[:3, :] / pts4D[3, :]
        
        return pts3D.T # Return as N x 3 array

# Code test
if __name__ == "__main__":
    print("📷 Testing Visual Odometry with dummy images...")
    
    # KITTI Camera Intrinsics (K matrix)
    # fx, fy = focal length | cx, cy = optical center
    K = np.array([[718.856, 0.0, 607.192],
                  [0.0, 718.856, 185.215],
                  [0.0, 0.0, 1.0]])
                  
    vo = VisualOdometry(K)
    
    # Create two fake frames
    img1 = np.zeros((375, 1242), dtype=np.uint8)
    cv2.rectangle(img1, (400, 100), (600, 300), 255, -1) # Square at x=400
    
    img2 = np.zeros((375, 1242), dtype=np.uint8)
    cv2.rectangle(img2, (450, 100), (650, 300), 255, -1) # Square moved to x=450
    
    # Extract features
    kp1, des1 = vo.extract_features(img1)
    kp2, des2 = vo.extract_features(img2)
    print(f"Features found -> Frame 1: {len(kp1)}, Frame 2: {len(kp2)}")
    
    # Match features
    matches = vo.match_features(des1, des2)
    print(f"Good matches found: {len(matches)}")
    
    if len(matches) > 8:
        # Estimate motion (Rotation and Translation)
        R, t, _, _, _ = vo.estimate_motion(kp1, kp2, matches)
        print("\n🔄 Estimated Rotation Matrix (R):\n", R)
        print("\n➡️ Estimated Translation Vector (t):\n", t)
    else:
        print("Not enough matches to compute motion.")