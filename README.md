# 📘 Neural Network Project on Fashion MNIST Dataset

This project demonstrates the implementation of a **Neural Network classifier** to recognize images from the **Fashion MNIST dataset** using TensorFlow and Keras.  
The Fashion MNIST dataset is a more challenging replacement for the classic MNIST digit dataset and contains grayscale images of 10 clothing categories.

---

## 📂 Project Structure

- **`Neural_Network_Project_MNIST_Fashion_Dataset.ipynb`**  
  Jupyter Notebook containing:
  - Data loading and preprocessing  
  - Model building and training  
  - Evaluation and visualization  

---

## 📊 Dataset Overview

- **Fashion MNIST**: 70,000 grayscale images (28×28 pixels) of 10 categories:
  - T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.
- **Training Set**: 60,000 images  
- **Test Set**: 10,000 images  
- Pixel values range from **0 to 255** (later scaled to **0–1** for training).

---

## ⚙️ Workflow

1. **Import Libraries**  
   TensorFlow, Keras, NumPy, Matplotlib.

2. **Load Dataset**  
   Directly available in `keras.datasets.fashion_mnist`.

3. **Preprocessing**  
   - Scale pixel values: divide by 255.0  
   - Split data: Training (55,000), Validation (5,000), Test (10,000).

4. **Visualization**  
   Display sample images using Matplotlib’s `imshow()`.

5. **Model Building**  
   - Input layer: Flatten (28×28 → 784 features)  
   - Hidden layers: Dense layers with ReLU activation  
   - Output layer: Dense(10) with softmax activation (for 10 categories).  

6. **Training**  
   - Optimizer: Adam/SGD  
   - Loss: Sparse Categorical Crossentropy  
   - Metrics: Accuracy  

7. **Evaluation**  
   - Test accuracy measurement  
   - Confusion matrix and classification report  
   - Visualization of predictions.

---

## 🧠 Key Concepts Explained

### 1. Pixel Normalization
- Original pixel values: 0–255.  
- Normalized by dividing by 255 → range 0–1.  
- Improves training stability and convergence.

### 2. One-Hot Encoding vs Sparse Labels
- Labels are integers `0–9`.  
- Instead of converting to one-hot vectors, Keras allows `SparseCategoricalCrossentropy`, which works directly with integer labels.

### 3. Activation Functions
- **ReLU (Rectified Linear Unit)**:  
  Helps in faster training and avoids vanishing gradient.  
- **Softmax** (output layer):  
  Converts raw scores into probability distribution over 10 classes.

### 4. Loss Function
- **Sparse Categorical Crossentropy**:  
  Measures the difference between predicted probabilities and actual class labels.

### 5. Optimizer
- **Adam**: Combines momentum & RMSProp for adaptive learning rates, generally faster than vanilla SGD.

### 6. Overfitting & Validation Set
- Training on entire dataset can cause overfitting.  
- A validation set (5,000 samples) helps monitor performance and tune hyperparameters.

### 7. Evaluation Metrics
- **Accuracy**: Percentage of correctly predicted images.  
- **Confusion Matrix**: Shows misclassifications across classes.  
- **Classification Report**: Precision, Recall, F1-score per class.

---

## 🚀 Results (Expected)

- Training Accuracy: ~90%  
- Test Accuracy: ~88–90% (depends on hyperparameters).  

---

## 🛠 Requirements

- Python 3.x  
- TensorFlow / Keras  
- NumPy  
- Matplotlib  

Install dependencies:
```bash
pip install tensorflow numpy matplotlib
