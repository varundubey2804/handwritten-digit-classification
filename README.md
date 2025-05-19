# handwritten-digit-classification
# MNIST Handwritten Digit Recognition with Custom Image Prediction

This project trains a neural network on the MNIST dataset using TensorFlow/Keras to classify handwritten digits (0–9). It includes functionality to capture and predict custom digits drawn or written on paper using a webcam (compatible with Google Colab).

---

##  Model Architecture

- Input: Flattened 28x28 grayscale images
- Hidden Layers:
  - Dense(128, activation='relu')
  - Dense(32, activation='relu')
- Output: Dense(10, activation='softmax') for 10 classes (digits 0-9)

---

##  Dataset

- **MNIST** dataset (built into Keras)
- Training size: 60,000 images
- Test size: 10,000 images

---

##  Training Details

- Loss: `sparse_categorical_crossentropy`
- Optimizer: `Adam`
- Epochs: 50
- Validation split: 20% of training data

---

##  Performance Visualization

Accuracy plots are generated for training and validation sets to monitor learning over epochs.

---

##  Custom Digit Prediction

### How it Works:
- Capture a photo using webcam (only in **Google Colab**).
- Preprocess:
  - Convert to grayscale
  - Resize to 28x28
  - Invert (if needed)
  - Normalize pixel values
- Predict digit using trained model.

---

##  Requirements

- Python
- TensorFlow
- NumPy
- Matplotlib
- PIL
- OpenCV
- Google Colab (for webcam capture)

---

##  How to Run (on Google Colab)

1. Upload your notebook to Google Colab.
2. Run all cells to train the model.
3. Use the webcam widget to capture a digit.
4. View predictions in the output.

---

## 📸 Sample Code for Webcam Capture (Colab)

```python
image_path = take_photo()
