# Gender Classification using ANN 🧠

A simple neural network-based desktop application to predict gender from biometric data using height, weight, and other physical attributes.

## 🔍 Overview

This project uses an Artificial Neural Network (ANN) built with **TensorFlow** and **Scikit-learn** to classify gender. The model takes in numeric features like height, weight, and shoe size and predicts whether the input corresponds to a male or female.

The frontend is built using **Tkinter** to make it user-friendly, with options for manual data entry and real-time model prediction display.

## 💡 Features

- Built a custom ANN with ReLU and Softmax activation layers
- Implemented end-to-end ML pipeline: data loading, preprocessing, model training, evaluation
- GUI frontend using Tkinter with:
  - Input fields for physical attributes
  - Real-time prediction output
- Evaluation metrics include:
  - Accuracy
  - Confusion Matrix
  - ROC Curve
  - AUC Score

## 🛠️ Tech Stack

- Python
- TensorFlow
- Scikit-learn
- Tkinter
- Matplotlib
- NumPy
- Pandas

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/samhitaprakash/new-ann-project-dental.git
   cd new-ann-project-dental
   pip install -r requirements.txt
   python ann_gui.py



 Sample Output
Model achieved high test accuracy and strong generalization, making it suitable for educational use and beginner-level ML experimentation.

Project Structure
graphql
Copy
Edit
├── ann_gui.py              # Main GUI script
├── model/                  # Contains ANN training code
├── data/                   # CSV dataset file
├── requirements.txt        # Python dependencies
├── README.md 

Author
Built with by Samhita Prakash
