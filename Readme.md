# Blood Cell Detection App

![Blood Cell Detection](path_to_sample_image.png)  
*A web-based application for detecting and classifying blood cells using YOLOv8.*

## 📌 Overview

The **Blood Cell Detection App** is a deep learning-powered web application that detects and classifies blood cells in microscopic images. It utilizes **YOLOv8** (You Only Look Once, version 8) to recognize three major types of blood cells:

- **RBC (Red Blood Cells)** - Erythrocytes
- **WBC (White Blood Cells)** - Leukocytes
- **Platelets (Thrombocytes)**

This application helps in analyzing blood samples, aiding researchers, medical professionals, and students in understanding and visualizing blood cell distribution.

---

## ✨ Features

✔️ **Upload Your Own Images** or use sample blood cell images provided in the app  
✔️ **Bounding Box Detection** with class labels and confidence scores  
✔️ **Tabular Representation** of detection results for easy analysis  
✔️ **Performance Metrics Display** including precision, recall, and confidence scores  
✔️ **User-Friendly Web Interface** built with Streamlit  
✔️ **Model Fine-tuned on BCCD Dataset** ensuring high accuracy  
✔️ **Download Processed Images** with annotations  

---

## 🚀 How to Use

### 🔹 1. Install Dependencies
Before running the app, install the required dependencies using the following command:
```bash
pip install -r requirements.txt
```

### 🔹 2. Run the Application
Start the Streamlit web application with:
```bash
streamlit run app.py
```

### 🔹 3. Upload & Detect
- Open the web application in your browser.
- Upload a microscopic blood cell image or use sample images provided.
- View the results with detected bounding boxes and confidence scores.
- Analyze detected cell types in the tabular output.

### 🔹 4. Model Performance
- The "Model Performance" tab shows evaluation metrics such as **precision, recall, and F1-score**.
- Helps in understanding how well the model performs across different cell types.

---

## 📊 Model Training Details

This application uses a **YOLOv8 object detection model**, which was fine-tuned on the **Blood Cell Count Dataset (BCCD)**.

🔹 **Training Process:**
- Dataset: [BCCD Dataset](https://github.com/Shenggan/BCCD_Dataset)
- Training split: **80% train, 20% validation**
- Target classes: **RBC, WBC, and Platelets**
- Model architecture: **YOLOv8 (Ultralytics)**

🔹 **Performance Metrics:**
| Metric  | RBC  | WBC  | Platelets |
|---------|------|------|-----------|
| Precision | XX% | XX%  | XX% |
| Recall    | XX% | XX%  | XX% |
| F1-score  | XX% | XX%  | XX% |

*(Replace `XX%` with actual results after evaluation)*

---

## 📂 Dataset Information

The **BCCD Dataset** is an open-source dataset for blood cell detection, widely used in medical image analysis research. It contains annotated microscopy images of blood cells labeled as:
- **RBC (Red Blood Cells)**
- **WBC (White Blood Cells)**
- **Platelets**

🔗 Dataset Link: [BCCD Dataset](https://github.com/Shenggan/BCCD_Dataset)

---

## 🏗️ Implementation Details

### 🛠️ Technologies Used
- **[YOLOv8](https://docs.ultralytics.com/)** - Deep learning-based object detection model
- **[Streamlit](https://streamlit.io/)** - Web framework for building interactive applications
- **Python, OpenCV, Pandas, NumPy** - Data processing and visualization

### 📌 Project Structure
```
📂 BloodCellDetectionApp
│── 📂 models          # Pre-trained YOLOv8 model
│── 📂 dataset         # Sample images and dataset-related files
│── 📂 src             # Core implementation scripts
│── app.py            # Main application script (Streamlit)
│── requirements.txt  # Python dependencies
│── README.md         # Documentation (This File)
```

---

## 🤝 Contributions & Support

👥 Contributions, issues, and feature requests are welcome! Feel free to open an issue or submit a pull request.

1. **Fork** the repository
2. **Clone** it to your local machine
3. **Create a new branch** for your feature/bugfix
4. **Commit your changes** and push
5. **Submit a Pull Request**

---

## 📜 License

This project is licensed under the **MIT License**. You are free to use, modify, and distribute it as long as proper credit is given.

---

## 📚 References

- **BCCD Dataset:** [https://github.com/Shenggan/BCCD_Dataset](https://github.com/Shenggan/BCCD_Dataset)
- **YOLOv8 Documentation:** [https://docs.ultralytics.com/](https://docs.ultralytics.com/)
- **Streamlit:** [https://streamlit.io/](https://streamlit.io/)

🚀 *Happy Coding! & Stay Curious!* 🧬

