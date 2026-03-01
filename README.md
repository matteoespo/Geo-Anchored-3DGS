# Geo-Anchored Visual SLAM & 3D Reconstruction

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)

An end-to-end Python pipeline for Monocular Visual Odometry, GNSS (GPS) sensor fusion, and sparse 3D point cloud generation. This project serves as the foundational tracking and mapping frontend for advanced 3D scene reconstruction techniques like **3D Gaussian Splatting**.

*(GIFs and Screenshots soon!)*

## Features

* **Visual Odometry**: Real-time frame-to-frame tracking using ORB features, feature matching (Brute-Force + Lowe's Ratio), and Essential Matrix pose recovery.
* **GNSS Integration**: Converts WGS84 global coordinates (Latitude, Longitude, Altitude) to local Cartesian coordinates (ENU) for accurate scaling and drift reduction.
* **3D Mapping**: Triangulates matched 2D points into a 3D coordinate space to generate a sparse point cloud of the environment.
* **Modular Architecture**: Clean, object-oriented Python design (`DataLoader`, `VisualOdometry`, `MapManager`).

## Project Structure

```text
Geo-Anchored-3DGS/
├── data/                   # KITTI Dataset sequence folder (ignored in git)
├── src/
│   ├── dataloader.py       # Handles images and WGS84 to local coordinate transforms
│   ├── visual_odometry.py  # ORB extraction, matching, and pose estimation
│   └── mapping.py          # Global trajectory tracking and 3D point cloud storage
├── main.py                 # Core SLAM loop and visualization
├── requirements.txt        # Python dependencies
└── README.md
```

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   cd YOUR_REPO_NAME
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Dataset Preparation:**
   * Download the [KITTI Odometry Dataset](http://www.cvlibs.net/datasets/kitti/eval_odometry.php).
   * Extract `sequence 00` into the `data/kitti_sequence_00` directory. Ensure `image_02` and `oxts` (if using Raw GPS data) are present.

4. **Run the SLAM Pipeline:**
   ```bash
   python main.py
   ```

## Roadmap / Next Steps

- [x] Basic Monocular Visual Odometry
- [x] GNSS/GPS Data Parsing and Coordinate Transformation
- [x] Stereo Triangulation for Sparse Point Cloud
- [ ] Sensor Fusion Optimization (GTSAM / Factor Graphs)
- [ ] Export trajectory and point cloud to COLMAP format
- [ ] Integration with 3D Gaussian Splatting for photorealistic rendering

## License
This project is open-source and available under the MIT License.
