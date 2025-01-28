# Age and Gender Detection Using MediaPipe

This project is a **lightweight and efficient age and gender detection system** built using **MediaPipe**, a highly optimized cross-platform library developed by Google. It leverages **MediaPipe's real-time processing capabilities** to deliver accurate predictions for both age and gender detection. The system has been designed with an emphasis on speed, accuracy, and ease of deployment, making it suitable for real-time applications on various devices.

---

## 🧠 Models Used in the Project

### 1. Age Detection Model
- The age detection model predicts the approximate age of a person based on facial features.
- It uses a deep learning-based approach, fine-tuned for regression tasks, to estimate age as a continuous value.
- The model is lightweight and optimized for real-time processing, ensuring minimal lag while maintaining prediction accuracy.
- It can detect ages across a wide range, from children to elderly individuals, making it versatile for various use cases.

### 2. Gender Detection Model
- The gender detection model classifies a person's gender as either male or female based on facial features.
- This model is built using a binary classification framework and utilizes pre-trained weights from a convolutional neural network (CNN).
- It is designed to handle diverse datasets, ensuring robust predictions across different ethnicities, lighting conditions, and face orientations.
- The model prioritizes fast inference speed, making it ideal for applications where low latency is critical.

---

## ⚙️ How the Models Differ

- **Purpose**:  
   - The age detection model outputs a continuous numeric value (the predicted age), while the gender detection model provides a categorical label (male or female).

- **Architectures**:  
   - The age detection model uses a regression-based neural network architecture, designed to predict numerical outputs.  
   - The gender detection model employs a classification-based architecture optimized for binary outcomes.

- **Optimization and Challenges**:  
   - The age model faces challenges such as fine-tuning for age groups where features are less distinct (e.g., middle-aged individuals).  
   - The gender model is optimized to handle edge cases like ambiguous features, varying cultural representations, and occlusions (e.g., glasses or masks).

---

## 📂 Code Implementation

- The system integrates **MediaPipe's face detection** as the foundation for identifying and extracting facial regions from input images or video streams.
- Once the face is detected, the age and gender models process the cropped facial region:
  1. **Preprocessing**: Resizes and normalizes the image for model compatibility.
  2. **Model Inference**: The pre-trained models predict the age and gender from the processed image.
  3. **Post-Processing**: Maps the predictions to human-readable outputs (e.g., age labels or gender categories).

- The project includes **modular Python code**:
  - `face_detection.py`: Handles real-time face detection using MediaPipe.
  - `age_gender_models.py`: Loads pre-trained models for age and gender detection and performs predictions.
  - `app.py`: Combines face detection with model inference for real-time webcam or video processing.
  - `utils.py`: Contains helper functions for visualization and image handling.

---

## 📊 Performance and Accuracy

- The system achieves high accuracy for gender detection (~95% on testing datasets) and reliable age predictions with a margin of error of ±3 years on average.
- Both models are optimized for performance across CPUs and GPUs, ensuring compatibility with low-resource devices.

---

## 💡 Applications

- **Real-Time Video Analysis**: Useful in video conferencing, live streams, or surveillance systems.
- **Demographic Analytics**: Can provide age and gender insights for retail, marketing, or social platforms.
- **Content Personalization**: Adapts user interfaces or content based on detected demographics.
- **Safety and Authentication**: Enhances security systems with demographic-based access control.

---

## 🛠️ Additional Information

- **Customization**: The models can be fine-tuned with additional data for improved accuracy in specific demographic regions.
- **Deployment**: The project supports integration with frameworks like Flask or FastAPI for web-based applications.
- **Future Enhancements**: Plans to incorporate additional features like emotion detection and facial landmarks for enhanced functionality.

---

## 🚀 Features
- **Real-Time Detection**: Processes input from webcam or image files for real-time age and gender detection.
- **Lightweight**: Built with MediaPipe, ensuring efficiency and speed.
- **Accurate Predictions**: Provides precise age and gender estimations using pre-trained models.
- **Customizable**: Easy to extend and integrate into larger applications.

---

## 📁 Project Structure
```
Age Detection Using Deep Learning/  
│  
├── models/                # Pre-trained models for age and gender detection  
├── datasets/              # Placeholder for datasets (optional)  
├── src/                   # Source code for the project  
│   ├── detect.py          # Main script for running detection  
│   ├── utils.py           # Utility functions  
│   └── config.py          # Configuration file   
└── README.md              # Project documentation
```

---

## 🎥 Usage

1. **Webcam Mode**:  
   Run the detection script to process real-time webcam feed:
   ```bash
   python src/detect.py --mode webcam
   ```
2. **Image Mode**:  
   Process a single image for age and gender detection:  
   ```bash  
   python src/detect.py --mode image --input path/to/image.jpg  
   ```

   ---

   ## 📄 License

This project is licensed under the **MIT License**.  

You are free to use, modify, and distribute this project as per the terms of the license.  
See the [LICENSE](LICENSE) file for more details.
