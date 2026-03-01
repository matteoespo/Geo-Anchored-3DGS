import cv2
import numpy as np
import os
import shutil

# Import custom modules
from src.dataloader import KittiDataloader
from src.visual_odometry import VisualOdometry
from src.mapping import MapManager

def create_dummy_kitti_for_main(base_path="data/dummy_kitti"):
    """Creates a temporary dummy dataset with moving shapes to test the pipeline."""
    os.makedirs(os.path.join(base_path, "image_02", "data"), exist_ok=True)
    os.makedirs(os.path.join(base_path, "oxts", "data"), exist_ok=True)
    
    for i in range(10):
        # Create a moving square
        img = np.zeros((375, 1242, 3), dtype=np.uint8)
        # Move square by 20 pixels each frame
        x_offset = 200 + i * 20 
        cv2.rectangle(img, (x_offset, 100), (x_offset + 200, 300), (255, 255, 255), -1)
        cv2.putText(img, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imwrite(os.path.join(base_path, "image_02", "data", f"{i:010d}.png"), img)
        
        # Fake GPS moving
        with open(os.path.join(base_path, "oxts", "data", f"{i:010d}.txt"), "w") as f:
            f.write(f"{49.0 + i*0.0001} {8.4 + i*0.0001} 112.0 0 0 0 0 0 0 0 0 0 0 0\n")

def main():
    print("Starting SLAM Pipeline...")
    
    # Prepare data
    dataset_path = "data/dummy_kitti"
    create_dummy_kitti_for_main(dataset_path)
    
    # Initialize Dataloader
    dataloader = KittiDataloader(dataset_path)
    print(f"Dataset loaded. Total frames: {len(dataloader)}")
    
    # Initialize Visual Odometry with KITTI Camera Intrinsics
    K = np.array([[718.856, 0.0, 607.192],
                  [0.0, 718.856, 185.215],
                  [0.0, 0.0, 1.0]])
    vo = VisualOdometry(K)
    mapper = MapManager()
    
    # Variables to store the previous frame's data
    prev_img = None
    prev_kp = None
    prev_desc = None
    
    # Main SLAM loop
    for i in range(len(dataloader)):
        # Read current frame and GPS
        curr_img, curr_gps = dataloader.get_frame(i)
        
        # Extract features for the current frame
        curr_kp, curr_desc = vo.extract_features(curr_img)
        
        # If it's the first frame, save it and continue
        if i == 0:
            prev_img = curr_img
            prev_kp = curr_kp
            prev_desc = curr_desc
            continue
            
        # Match features between previous and current frame
        matches = vo.match_features(prev_desc, curr_desc)
        
        print(f"\n--- Frame {i} ---")
        print(f"GPS Position (Local X,Y,Z): [{curr_gps[0]:.2f}, {curr_gps[1]:.2f}, {curr_gps[2]:.2f}]")
        
        if len(matches) > 8:
            # Estimate Visual Motion (and get the matched 2D points 'pts1' and 'pts2')
            R, t, pts1, pts2, mask = vo.estimate_motion(prev_kp, curr_kp, matches)
            
            # Save the PREVIOUS global pose before updating it
            prev_global_pose = mapper.current_pose.copy()
            
            # Update Global Trajectory
            global_pos = mapper.update_pose(R, t)
            print(f"Global Camera Pose: X={global_pos[0]:.2f}, Y={global_pos[1]:.2f}, Z={global_pos[2]:.2f}")
            
            # Triangulate and Add Points to Map
            # Filter the points using the RANSAC mask to only triangulate the "good" ones
            good_pts1 = pts1[mask.ravel() == 1]
            good_pts2 = pts2[mask.ravel() == 1]
            
            if len(good_pts1) > 0:
                points_3d = vo.triangulate_points(R, t, good_pts1, good_pts2)
                mapper.add_points(points_3d, prev_global_pose)
                print(f"☁️ Added {len(points_3d)} 3D points to the map.")
            
            # VISUALIZATION
            # Draw lines connecting the matched points to see the tracking
            display_img = cv2.drawMatches(prev_img, prev_kp, curr_img, curr_kp, matches[:50], None, 
                                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            
            # Resize for better viewing on screen
            display_img = cv2.resize(display_img, (1200, 400))
            cv2.imshow("Visual Odometry Tracking", display_img)
            
            # Press 'q' to quit early, or wait 500ms for the next frame
            if cv2.waitKey(500) & 0xFF == ord('q'):
                break
        else:
            print("Not enough matches to compute visual odometry.")
            
        # Update previous frame variables for the next iteration
        prev_img = curr_img
        prev_kp = curr_kp
        prev_desc = curr_desc

    # Show the final 3D trajectory and map
    print("Plotting trajectory and point cloud...")
    print(f"Total number of points in the PC: {len(mapper.point_cloud)}")
    mapper.plot_trajectory_and_map()
    
    # Clean up OpenCV windows and dummy data
    cv2.destroyAllWindows()
    shutil.rmtree(dataset_path)
    print("\nPipeline finished!")

if __name__ == "__main__":
    main()