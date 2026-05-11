# 🛡️ Robust Scene Classifier

A production-grade machine learning application designed to classify natural landscapes (buildings, forests, glaciers, mountains, sea, and streets) with high accuracy, while employing enterprise-level input validation and guardrails.

## 🌟 Overview

The **Robust Scene Classifier** uses an advanced transfer-learning architecture (EfficientNetV2 base) to achieve **~92.3% accuracy** on environmental classifications. 

What sets this project apart is its **Industrial Guard** system: it does not just blindly classify images. It calculates image complexity, brightness thresholds, and standard deviations to actively reject "garbage" data such as flat drawings, text documents, or overly confused inputs, preventing hallucinated predictions.

## 🚀 Features

*   **Microservice Architecture**: Decoupled FastAPI backend and Streamlit frontend.
*   **Intelligent Input Validation**: Employs mathematical **Shannon Entropy** to measure pixel "chaos," actively rejecting non-landscape images (sketches/documents) regardless of JPEG noise.
*   **Confidence Guardrails**: Calculates classification confidence gaps to flag ambiguous predictions (e.g., Mountain vs. Glacier).
*   **Adaptive UI**: Intelligently handles native image resolutions, displaying small dataset images in crisp "Polaroid" containers and HD photos at full width using lossless Base64 HTML rendering.
*   **Dynamic Configuration**: Cloud-ready path routing utilizing environment variables (`ML_WORK_DIR`) with automatic cross-platform fallback mechanisms.

## 🛠️ Tech Stack

*   **Backend**: FastAPI, Uvicorn
*   **Frontend**: Streamlit, HTML/CSS
*   **Machine Learning**: TensorFlow, Keras, Pillow
*   **Data Processing**: Pandas, NumPy

## 💻 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Alishba-Hamid258/Robust-Scene-Classifier.git
   cd Robust-Scene-Classifier
   ```

2. **Create a virtual environment & install dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Set up the Model:**
   Place the `best_model.h5` inside a `models/` directory. (If using a custom directory, set the `ML_WORK_DIR` environment variable).

4. **Run the Backend API:**
   ```bash
   python app.py
   # The API will be available at http://localhost:8000
   ```

5. **Run the Frontend UI:**
   ```bash
   streamlit run streamlit_app.py
   # The Web UI will be available at http://localhost:8501
   ```

## 🧠 Model Architecture & Training

The active model utilizes an **EfficientNetV2S** base, trained using a rigorous two-phase transfer learning approach:
1. Feature extraction with a frozen base model.
2. Gentle fine-tuning with learning rate reduction on plateaus.
*Data Augmentation (rotation, shifting, flipping) was heavily utilized to prevent overfitting on the natural landscape dataset.*
