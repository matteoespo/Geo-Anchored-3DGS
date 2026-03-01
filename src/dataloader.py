import os
import glob
import numpy as np
import cv2
from pyproj import Transformer

class KittiDataloader:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        
        # Paths to KITTI specific folders
        self.image_dir = os.path.join(dataset_path, "image_02", "data")
        self.oxts_dir = os.path.join(dataset_path, "oxts", "data")
        
        # Get sorted lists of files
        self.image_files = sorted(glob.glob(os.path.join(self.image_dir, "*.png")))
        self.oxts_files = sorted(glob.glob(os.path.join(self.oxts_dir, "*.txt")))
        
        if len(self.image_files) == 0 or len(self.oxts_files) == 0:
            print(f"Warning: No data found in {dataset_path}. Check your dataset path.")
        elif len(self.image_files) != len(self.oxts_files):
            print("Warning: Number of images and GPS files do not match!")
            
        self.origin_lat_lon_alt = None
        self.transformer = Transformer.from_crs("epsg:4326", "epsg:4978", always_xy=True)
        self.origin_ecef = None

    def __len__(self):
        """Returns the total number of frames in the sequence."""
        return min(len(self.image_files), len(self.oxts_files))

    def set_origin(self, lat: float, lon: float, alt: float):
        """Sets the very first GPS point as the origin (0,0,0) of our 3D map."""
        self.origin_lat_lon_alt = (lat, lon, alt)
        self.origin_ecef = self.transformer.transform(lon, lat, alt)
        print(f"Origin set to: Lat {lat}, Lon {lon}, Alt {alt}")

    def wgs84_to_local(self, lat: float, lon: float, alt: float) -> np.ndarray:
        """Converts a GPS coordinate to (X, Y, Z) in meters relative to the origin."""
        if self.origin_ecef is None:
            raise ValueError("You must call set_origin() with the first GPS point before converting!")
        
        current_ecef = self.transformer.transform(lon, lat, alt)
        dx = current_ecef[0] - self.origin_ecef[0]
        dy = current_ecef[1] - self.origin_ecef[1]
        dz = current_ecef[2] - self.origin_ecef[2]
        
        return np.array([dx, dy, dz])

    def load_image(self, frame_index: int) -> np.ndarray:
        """Loads an image from the dataset using OpenCV."""
        img_path = self.image_files[frame_index]
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        return img

    def load_gps_data(self, frame_index: int):
        """Reads the oxts text file and extracts Lat, Lon, Alt."""
        oxts_path = self.oxts_files[frame_index]
        with open(oxts_path, 'r') as f:
            # Read the first line, split by spaces
            line = f.readline().strip().split()
            # KITTI format: lat, lon, alt are the first 3 values
            lat, lon, alt = float(line[0]), float(line[1]), float(line[2])
        return lat, lon, alt

    def get_frame(self, frame_index: int):
        """Core function: Returns the image and its local (X, Y, Z) position in meters."""
        img = self.load_image(frame_index)
        lat, lon, alt = self.load_gps_data(frame_index)
        
        # If it's the very first frame, automatically set it as the origin
        if self.origin_ecef is None:
            self.set_origin(lat, lon, alt)
            
        local_xyz = self.wgs84_to_local(lat, lon, alt)
        
        return img, local_xyz


# Code test
if __name__ == "__main__":
    import shutil
    
    print("Creating dummy KITTI dataset for testing...")
    base_path = "data/dummy_kitti"
    os.makedirs(os.path.join(base_path, "image_02", "data"), exist_ok=True)
    os.makedirs(os.path.join(base_path, "oxts", "data"), exist_ok=True)
    
    # Create 3 fake frames
    for i in range(3):
        # 1. Create a fake black image (1242x375, typical KITTI size)
        dummy_img = np.zeros((375, 1242, 3), dtype=np.uint8)
        cv2.putText(dummy_img, f"Frame {i}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 3)
        cv2.imwrite(os.path.join(base_path, "image_02", "data", f"{i:010d}.png"), dummy_img)
        
        # 2. Create a fake GPS text file (moving slightly North-East)
        with open(os.path.join(base_path, "oxts", "data", f"{i:010d}.txt"), "w") as f:
            f.write(f"{49.0 + i*0.0001} {8.4 + i*0.0001} 112.0 0 0 0 0 0 0 0 0 0 0 0\n")

    print("Dummy dataset created!\n")
    
    # --- Actual Test ---
    print("Initializing Dataloader...")
    loader = KittiDataloader(base_path)
    print(f"Total frames found: {len(loader)}")
    
    for i in range(len(loader)):
        image, position = loader.get_frame(i)
        print(f"Frame {i}: Position (X, Y, Z) meters = [{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}]")
        print(f"         Image shape = {image.shape}")

    # Clean up dummy data after test
    shutil.rmtree(base_path)