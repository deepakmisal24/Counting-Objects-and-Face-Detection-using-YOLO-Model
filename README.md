# 🧠 Counting Objects and Face Detection using OpenCV

This project uses **YOLOv8** and **OpenCV** to detect and count objects, specifically focusing on identifying people and then detecting **faces** within the detected persons. It is implemented in **Google Colab** and leverages both **Ultralytics YOLO** and **Haar Cascades**.

## 🔍 Features

* Upload and analyze an image using `Google Colab`.
* Detect all objects using the YOLOv8 model.
* Highlight and count the number of **persons** detected.
* Perform **face detection** on each person using OpenCV's Haar cascade classifier.
* Save the annotated image with bounding boxes and labels.
* Display the final output image in the notebook.

## 📸 Demo

> ![Sample Output](output_with_boxes)

## 🛠️ Technologies Used

* **Python**
* **OpenCV**
* **Ultralytics YOLOv8 (`yolov8n.pt`)**
* **Google Colab**
* **Haar Cascade Classifier**

## 📦 Requirements

To run this project in Google Colab:

```bash
pip install ultralytics opencv-python
```

No manual installation of `cv2` is needed in Colab — it is preinstalled.

## 📁 Project Structure

```
.
├── counting_objects_face_detection.ipynb
├── output_with_boxes.jpg
└── README.md
```

## 🚀 How It Works

1. **Upload an image** in Google Colab.
2. **YOLOv8** detects all objects in the image.
3. Each detected object is labeled and counted.
4. If a detected object is a **person**, it is passed to the **Haar Cascade** classifier.
5. **Faces** are detected inside the person region.
6. Output is saved as `output_with_boxes.jpg` with all detections highlighted.

## 🔎 Sample Code Snippet

```python
model = YOLO("yolov8n.pt")
results = model(image)
detections = results[0]
total_objects = len(detections.boxes)

# Face detection on detected 'person' boxes
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
```

## 💡 Future Improvements

* Switch to deep learning-based face detection for better accuracy.
* Add support for real-time webcam detection.
* Integrate a simple web UI using Gradio or Streamlit.
* Count and visualize other specific objects (e.g., vehicles, animals).

## 👤 Author

**Deepak Misal**
🔗 [LinkedIn Profile](https://www.linkedin.com/in/deepakmisal24/)

---

Let me know if you'd like a version with Colab badges or want to auto-generate the sample image from your code output.
