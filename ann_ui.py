import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt

class ANNApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ANN Training UI")
        
        # UI Elements
        tk.Label(root, text="Artificial Neural Network for Gender Classification", font=("Arial", 14)).pack(pady=10)
        self.load_button = tk.Button(root, text="Load Dataset", command=self.load_data)
        self.load_button.pack(pady=5)
        
        self.train_button = tk.Button(root, text="Train Model", command=self.train_model, state=tk.DISABLED)
        self.train_button.pack(pady=5)
        
        self.test_button = tk.Button(root, text="Evaluate Model", command=self.evaluate_model, state=tk.DISABLED)
        self.test_button.pack(pady=5)
        
        self.plot_button = tk.Button(root, text="Generate Graphs", command=self.generate_graphs, state=tk.DISABLED)
        self.plot_button.pack(pady=5)

        self.predict_button = tk.Button(root, text="Manual Prediction", command=self.manual_prediction, state=tk.DISABLED)
        self.predict_button.pack(pady=5)

        self.equation_button = tk.Button(root, text="Display Equation", command=self.display_equation, state=tk.DISABLED)
        self.equation_button.pack(pady=5)
        
        self.status_label = tk.Label(root, text="", font=("Arial", 12))
        self.status_label.pack(pady=10)
        
        # Data and Model Variables
        self.data = None
        self.model = None
        self.X_test = None
        self.y_test = None
        self.scaler = None
        self.history = None

    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            try:
                # Load dataset
                self.data = pd.read_csv(file_path)
                self.status_label.config(text="Dataset Loaded Successfully!")
                self.train_button.config(state=tk.NORMAL)
            except Exception as e:
                messagebox.showerror("Error", f"Could not load file: {e}")

    def train_model(self):
        if self.data is None:
            messagebox.showerror("Error", "No dataset loaded!")
            return
        
        try:
            # Prepare data
            X = self.data.iloc[:, :-1].values
            y = self.data.iloc[:, -1].values
            y = to_categorical(y)  # Convert labels to one-hot encoding
            
            # Split data
            X_train, self.X_test, y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Standardize data
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            self.X_test = self.scaler.transform(self.X_test)
            
            # Create ANN model
            self.model = Sequential([
                Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
                Dense(32, activation='relu'),
                Dense(2, activation='softmax')  # Output layer for binary classification
            ])
            self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            
            # Train model
            self.history = self.model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)
            self.status_label.config(text="Model Trained Successfully!")
            self.test_button.config(state=tk.NORMAL)
            self.plot_button.config(state=tk.NORMAL)
            self.predict_button.config(state=tk.NORMAL)
            self.equation_button.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Error", f"Training failed: {e}")

    def manual_prediction(self):
        if self.model is None or self.scaler is None:
            messagebox.showerror("Error", "Model not trained yet!")
            return
        
        # Create a new window for manual input
        def predict():
            try:
                # Get manual input
                values = [float(entry.get()) for entry in inputs]
                values = np.array(values).reshape(1, -1)
                values = self.scaler.transform(values)  # Scale the inputs
                
                # Make prediction
                prediction = self.model.predict(values)
                predicted_class = np.argmax(prediction)
                result = "Male" if predicted_class == 0 else "Female"
                messagebox.showinfo("Prediction Result", f"The predicted gender is: {result}")
            except Exception as e:
                messagebox.showerror("Error", f"Prediction failed: {e}")

        manual_window = tk.Toplevel(self.root)
        manual_window.title("Manual Prediction")
        tk.Label(manual_window, text="Enter morphometric parameter values:", font=("Arial", 12)).pack(pady=10)
        
        # Feature names
        feature_names = [
            "Maximum Ramus Breadth (MRB)",
            "Bi-condylar Width (BiCW)",
            "Condylar Height (CoH)",
            "Coronoid Height (CorH)",
            "Bigonial Width (BiGW)",
            "Bimental Width (BiMW)",
            "Gonial Angle (GoA)"
        ]
        
        inputs = []
        for feature in feature_names:
            frame = tk.Frame(manual_window)
            frame.pack(pady=5)
            tk.Label(frame, text=f"{feature}:").pack(side=tk.LEFT)
            entry = tk.Entry(frame)
            entry.pack(side=tk.RIGHT)
            inputs.append(entry)
        
        tk.Button(manual_window, text="Predict", command=predict).pack(pady=10)

    def display_equation(self):
        if self.model is None:
            messagebox.showerror("Error", "Model not trained yet!")
            return
        
        try:
            # Extract weights and biases
            equation = ""
            for i, layer in enumerate(self.model.layers):
                weights, biases = layer.get_weights()
                equation += f"Layer {i+1}:\n"
                for neuron in range(len(biases)):
                    neuron_eq = f"Neuron {neuron}: "
                    neuron_eq += " + ".join([f"({w:.4f} * x{j})" for j, w in enumerate(weights[:, neuron])])
                    neuron_eq += f" + ({biases[neuron]:.4f})"
                    equation += neuron_eq + "\n"
                equation += "\n"

            # Show equation in a pop-up window
            eq_window = tk.Toplevel(self.root)
            eq_window.title("Model Equation")
            text_widget = tk.Text(eq_window, wrap=tk.WORD, height=20, width=80)
            text_widget.pack(pady=10)
            text_widget.insert(tk.END, equation)
            text_widget.config(state=tk.DISABLED)  # Make read-only
        except Exception as e:
            messagebox.showerror("Error", f"Could not display equation: {e}")

    def evaluate_model(self):
        if self.model is None or self.X_test is None or self.y_test is None:
            messagebox.showerror("Error", "Model not trained yet!")
            return
        
        try:
            # Evaluate model
            loss, accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=0)
            messagebox.showinfo("Model Performance", f"Test Accuracy: {accuracy:.2f}\nTest Loss: {loss:.2f}")
        except Exception as e:
            messagebox.showerror("Error", f"Evaluation failed: {e}")

    def generate_graphs(self):
        if self.history is None:
            messagebox.showerror("Error", "No training history found!")
            return

        try:
            # Plot Accuracy and Loss Curves
            fig, axs = plt.subplots(1, 2, figsize=(12, 5))

            # Accuracy
            axs[0].plot(self.history.history['accuracy'], label='Training Accuracy')
            axs[0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
            axs[0].set_title('Model Accuracy')
            axs[0].set_xlabel('Epoch')
            axs[0].set_ylabel('Accuracy')
            axs[0].legend()

            # Loss
            axs[1].plot(self.history.history['loss'], label='Training Loss')
            axs[1].plot(self.history.history['val_loss'], label='Validation Loss')
            axs[1].set_title('Model Loss')
            axs[1].set_xlabel('Epoch')
            axs[1].set_ylabel('Loss')
            axs[1].legend()

            plt.tight_layout()
            plt.show()
        except Exception as e:
            messagebox.showerror("Error", f"Graph generation failed: {e}")

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = ANNApp(root)
    root.mainloop()
