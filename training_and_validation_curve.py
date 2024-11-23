import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import pickle

# Load Dataset
def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

# Build Model
def build_model(input_shape):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dense(32, activation='relu'),
        Dense(2, activation='softmax')  # Output layer for binary classification
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train Model
def train_model(model, X_train, y_train, epochs=20, batch_size=32, validation_split=0.2):
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=1)
    return history

# Save Training History
def save_history(history, filename='history.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(history.history, f)

# Load Training History
def load_history(filename='history.pkl'):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Plot Training and Validation Curves
def plot_training_curves(history):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Accuracy
    axs[0].plot(history['accuracy'], label='Training Accuracy')
    axs[0].plot(history['val_accuracy'], label='Validation Accuracy')
    axs[0].set_title('Model Accuracy')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend()

    # Loss
    axs[1].plot(history['loss'], label='Training Loss')
    axs[1].plot(history['val_loss'], label='Validation Loss')
    axs[1].set_title('Model Loss')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Loss')
    axs[1].legend()

    plt.tight_layout()
    plt.show()

# Plot Confusion Matrix
def plot_confusion_matrix(y_true, y_pred_classes, labels):
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Plot ROC Curve
def plot_roc_curve(y_true, y_pred_prob, classes):
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob[:, 1])  # Assuming class index 1 corresponds to "Positive"
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.show()

# Print Classification Report
def print_classification_report(y_true, y_pred_classes, labels):
    report = classification_report(y_true, y_pred_classes, target_names=labels, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    print(report_df)

# Main Function
def main():
    # Load the dataset
    data = load_data('./newdataset.csv')  # Replace with your dataset path

    # Preprocess the data
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    y = to_categorical(y)  # Convert labels to one-hot encoding

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Build the model
    model = build_model(X_train.shape[1])

    # Train the model
    history = train_model(model, X_train, y_train)

    # Save the training history
    save_history(history)

    # Load the training history for plotting
    loaded_history = load_history()

    # Plot training and validation curves
    plot_training_curves(loaded_history)

    # Evaluate the model
    y_pred_prob = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred_classes, labels=['Male', 'Female'])

    # Plot ROC curve
    plot_roc_curve(y_true, y_pred_prob, classes=['Male', 'Female'])

    # Print classification report
    print_classification_report(y_true, y_pred_classes, labels=['Male', 'Female'])

# Execute the main function
if __name__ == "__main__":
    main()
