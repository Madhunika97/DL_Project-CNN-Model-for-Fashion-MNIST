# DL_Project-CNN-Model-for-Fashion-MNIST

# ** Detailed Explanation of the CNN Model for Fashion-MNIST**
This Python script is designed to classify images from the **Fashion-MNIST dataset** using a **Convolutional Neural Network (CNN)** implemented in TensorFlow/Keras. The dataset consists of **grayscale images (28x28 pixels)** of 10 different types of fashion items. The model will be trained to recognize these items and make predictions.  

---

## **1️⃣ Import Required Libraries**
```python
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
```
### **🔹 What This Does:**
- `pandas` → Used for loading and manipulating CSV files (dataset).  
- `numpy` → Supports array operations and numerical calculations.  
- `tensorflow` / `keras` → TensorFlow’s deep learning framework, used for building and training the CNN model.  
- `matplotlib.pyplot` → Used for plotting and displaying images.  

---

## **2️⃣ Load Dataset from CSV Files**
```python
train_file = "fashion-mnist_train.csv"  # Update path if needed
test_file = "fashion-mnist_test.csv"    # Update path if needed

df_train = pd.read_csv(train_file)
df_test = pd.read_csv(test_file)
```
### **🔹 What This Does:**
- Loads the **Fashion-MNIST dataset** stored in CSV format.
- `df_train` → Training dataset (used for model learning).
- `df_test` → Test dataset (used to evaluate performance).

---

## **3️⃣ Extract Labels and Pixel Data**
```python
y_train = df_train.iloc[:, 0].values  # First column contains labels
X_train = df_train.iloc[:, 1:].values  # Remaining columns contain pixel values

y_test = df_test.iloc[:, 0].values
X_test = df_test.iloc[:, 1:].values
```
### **🔹 What This Does:**
- `y_train` & `y_test` → Stores the **labels** (0-9) indicating the clothing category.  
- `X_train` & `X_test` → Stores the **pixel values** of images.  

📌 **Example of Data Structure (First Few Rows of `df_train`)**
| Label | Pixel1 | Pixel2 | ... | Pixel784 |
|--------|--------|--------|----|-----------|
| 3      |  0     |  34    | ... | 123       |
| 5      |  23    |  56    | ... | 87        |

- The **label** column represents the category of the clothing item.  
- The **pixel values** range from **0 (black)** to **255 (white)**.

---

## **4️⃣ Normalize Pixel Values**
```python
X_train = X_train / 255.0
X_test = X_test / 255.0
```
### **🔹 What This Does:**
- Divides all pixel values by **255** to scale them between **0 and 1**.
- Normalization improves **model performance** by ensuring all features have the same scale.

📌 **Example Before & After Normalization**
| Pixel1 (Original) | Pixel1 (Normalized) |
|------------------|-------------------|
| 0               | 0.0               |
| 128             | 0.5               |
| 255             | 1.0               |

---

## **5️⃣ Reshape Data for CNN**
```python
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)
```
### **🔹 What This Does:**
- Converts **1D arrays (784 pixels)** into **2D 28x28 grayscale images** with a single color channel (`1`).
- `-1` → Automatically adjusts the number of samples.

📌 **Shape Before & After Reshaping**
| Shape (Before) | Shape (After) |
|--------------|--------------|
| (60000, 784) | (60000, 28, 28, 1) |

---

## **6️⃣ Determine the Number of Classes**
```python
num_classes = len(np.unique(y_train))
print(f"Number of Classes: {num_classes}")
```
### **🔹 What This Does:**
- Uses `np.unique(y_train)` to find all **unique labels** (0-9).
- Prints the number of classes **(10 for Fashion-MNIST)**.

📌 **Fashion-MNIST Class Labels**
| Label | Clothing Item |
|------|---------------|
| 0    | T-shirt/top |
| 1    | Trouser |
| 2    | Pullover |
| 3    | Dress |
| 4    | Coat |
| 5    | Sandal |
| 6    | Shirt |
| 7    | Sneaker |
| 8    | Bag |
| 9    | Ankle boot |

---

## **7️⃣ Build the CNN Model**
```python
model = keras.Sequential([
    keras.layers.Input(shape=(28,28,1)),  # Input layer

    # Convolutional Layer 1
    keras.layers.Conv2D(32, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2,2),

    # Convolutional Layer 2
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2,2),

    # Fully Connected Layers
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),

    # Output Layer
    keras.layers.Dense(num_classes, activation='softmax')  # 10 categories
])
```
### **🔹 CNN Model Structure**
| Layer | Type | Output Shape |
|---------|----------------------|--------------|
| Conv2D  | 32 filters, 3x3 kernel | (26, 26, 32) |
| MaxPooling2D | 2x2 pooling | (13, 13, 32) |
| Conv2D  | 64 filters, 3x3 kernel | (11, 11, 64) |
| MaxPooling2D | 2x2 pooling | (5, 5, 64) |
| Flatten  | Converts to 1D | (1600) |
| Dense  | Fully Connected (128 neurons) | (128) |
| Dense  | Output Layer (10 classes) | (10) |

---

## **8️⃣ Compile the Model**
```python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```
### **🔹 What This Does:**
- **Optimizer:** `adam` → Adjusts weights for best performance.
- **Loss Function:** `sparse_categorical_crossentropy` → Used for multi-class classification.
- **Metric:** `accuracy` → Evaluates model performance.

---

## **9️⃣ Train the Model**
```python
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
```
### **🔹 What This Does:**
- **Trains** the CNN on the **training dataset**.
- Runs for **5 epochs** (can be adjusted).
- Uses `validation_data` to monitor performance.

---

## **🔟 Evaluate Model Performance**
```python
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc}")
```
### **🔹 What This Does:**
- Evaluates model performance on the **test set**.
- Prints the **test accuracy**.

---

## **1️⃣1️⃣ Predict a Sample Image**
```python
plt.imshow(X_test[0].reshape(28,28), cmap='gray')
plt.show()

prediction = model.predict(np.expand_dims(X_test[0], axis=0))
predicted_label = np.argmax(prediction)
print(f"\033[1mPredicted Label: {predicted_label} ({class_names[predicted_label]})\033[0m")
```
### **🔹 What This Does:**
- **Displays** a sample test image.
- Uses `model.predict()` to classify it.
- Prints the **predicted label** in **bold**.

---

# **🎯 Summary**
✅ Loaded Fashion-MNIST dataset  
✅ Preprocessed images (scaling & reshaping)  
✅ Built & trained a CNN model  
✅ Evaluated accuracy  
✅ Made predictions  
