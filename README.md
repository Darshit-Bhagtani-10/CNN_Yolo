# 🎯 TargetVision: Real-Time Object Detection with YOLO + CNN

**TargetVision** is a computer vision project that leverages the speed of **YOLO (You Only Look Once)** and the accuracy of **Convolutional Neural Networks (CNNs)** to perform object detection in real-time. Whether you're analyzing CCTV footage, detecting vehicles, or building smart surveillance systems — TargetVision gives you a modular base to get started fast.

---

## 🧠 Core Features

🔍 **YOLO-Based Object Detection**: Integrates YOLOv4-tiny for lightning-fast detection.

🧬 **CNN Classification Layer**: Classifies objects detected by YOLO with a secondary CNN model for added accuracy or category refinement.

🧠 **Two-Stage Processing Pipeline**: YOLO handles bounding box detection, and the CNN classifies individual crops.

📸 **Live Video / Webcam Support**: Runs inference on live camera feed or pre-recorded video files.

📁 **Dataset Support**: Structured to support your own labeled data for both YOLO and CNN fine-tuning.

💾 **Model Checkpoints**: Pre-trained weights loading and saving for fast testing and reuse.

🛠️ **Modular Codebase**: Easy to plug-and-play with your own models or modify logic.

---

## 🔧 Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| OpenCV | Video capture and image processing |
| TensorFlow / Keras | CNN modeling |
| YOLOv4-tiny | Real-time object detection |
| NumPy | Numerical operations |
| OS, Time, Glob | File system & preprocessing utilities |

---


---

## 🚀 How to Run

### 1. Clone the Repo

```bash
git clone https://github.com/your-username/TargetVision.git
cd CNN_Yolo-main

2. Create a Virtual Environment (Optional)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

3. Install Dependencies
pip install -r requirements.txt

4. Run YOLO Detection on Video
cd yolo-detection
python yolo_detect.py --source path_to_video.mp4 --weights yolov4-tiny.weights

5. Run CNN Classification on YOLO Outputs
cd cnn-classifier
python predict.py --input ../output_crops/

📈 Example Use Cases
Surveillance & Security

Traffic Analysis

Sports Player Detection

Smart Retail (detect people, items)

Industrial Quality Check


