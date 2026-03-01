import numpy as np
import matplotlib.pyplot as plt

class MapManager:
    def __init__(self):
        self.current_pose = np.eye(4)
        self.trajectory = [np.array([0.0, 0.0, 0.0])]
        
        # List to store 3D point cloud
        self.point_cloud = []
        
    def update_pose(self, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        T_relative = np.eye(4)
        T_relative[:3, :3] = R
        T_relative[:3, 3] = t.flatten()
        
        self.current_pose = self.current_pose @ T_relative
        global_t = self.current_pose[:3, 3]
        self.trajectory.append(global_t.copy())
        
        return global_t

    def add_points(self, local_points_3d: np.ndarray, reference_pose: np.ndarray):
        """
        Transforms local 3D points to the global coordinate system and stores them.
        """
        num_points = local_points_3d.shape[0]
        if num_points == 0:
            return

        # Convert N x 3 to 4 x N homogeneous coordinates
        ones = np.ones((num_points, 1))
        points_4d = np.hstack((local_points_3d, ones)).T 

        # Global transformation matrix of the reference camera
        global_points_4d = reference_pose @ points_4d

        # Convert back to 3D
        global_points_3d = global_points_4d[:3, :].T 

       # Filter out crazy outliers
        for pt in global_points_3d:
            if np.linalg.norm(pt) < 5000.0:
                self.point_cloud.append(pt)
        
    def plot_trajectory_and_map(self):
        """Plots the trajectory and the 3D point cloud."""
        traj = np.array(self.trajectory)
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot Trajectory
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], label='Camera Trajectory', color='b', linewidth=2)
        ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], color='g', marker='o', s=100, label='Start')
        
        # Plot Point Cloud (if we have points)
        if len(self.point_cloud) > 0:
            pc = np.array(self.point_cloud)
            ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], color='r', s=20, alpha=0.8, label='Map Points')
        
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_zlabel('Z (meters)')
        ax.set_title('3D SLAM Map and Trajectory')
        ax.legend()
        plt.show()