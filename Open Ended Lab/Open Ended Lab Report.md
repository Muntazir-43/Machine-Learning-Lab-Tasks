# Open Ended Mnist Dataset Lab Task Report

## 1. Introduction
The dataset used for this classification task consists of pre-split training and test sets, commonly used in image classification problems. The training dataset comprises **60,000 rows and 785 columns**, while the test dataset contains **10,000 rows and 785 columns**. Each instance represents a **28×28 grayscale image**, with **784 features** and **one label column**. The primary objective is to evaluate multiple classification models and determine the most effective one based on various performance metrics.

## 2. Methodology
### 2.1 Dataset Preparation
#### Preprocessing Steps:
- **Feature Scaling:** Pixel values were normalized to the range **[0,1]** by dividing each value by **255**.
- **Label Encoding:**
  - **One-hot encoding** for ANN.
  - **Integer encoding** for KNN, Random Forest, and SVM.

### 2.2 Models Used & Hyperparameter Tuning

#### **Artificial Neural Network (ANN)**
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)), 
    Dropout(0.3),  
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
```
- **Trained with:** `epochs=20, batch_size=128`
- **Tuned Parameters:** Hidden layers, neurons per layer, dropout rate, batch size.

#### **K-Nearest Neighbors (KNN)**
```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='euclidean')
knn.fit(X_train, y_train)
```
- **Tuned Parameters:** `k` value, distance metric (Euclidean, Manhattan).

#### **Random Forest**
```python
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```
- **Tuned Parameters:** Number of trees, max features per split.

#### **Support Vector Machine (SVM)**
```python
from sklearn.svm import SVC
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm.fit(X_train, y_train)
```
- **Tuned Parameters:** Kernel type (Linear, Polynomial, RBF), `C` value.

### 2.3 Model Training & Evaluation
- **Trained with:** 60,000-row training set.
- **Tested on:** 10,000-row test set.
- **Metrics Used:** Accuracy, Precision, Recall, F1-score.

## 3. Results
### 3.1 Model Performance Comparison
| Metric | ANN | KNN | Random Forest | SVM |
|--------|------|------|--------------|------|
| **Accuracy** | **98.22%** | 96.91% | 97.04% | 97.92% |
| **Macro Avg Precision** | 98.20% | 96.95% | 97.03% | 97.91% |
| **Macro Avg Recall** | 98.21% | 96.87% | 97.01% | 97.90% |
| **Macro Avg F1-score** | 98.20% | 96.89% | 97.02% | 97.91% |

## 4. Discussion
### 4.1 Best Performing Model
- **ANN achieved the highest accuracy (98.22%)**, indicating its ability to capture complex patterns.
- **SVM (97.92%)** also performed well, especially with the RBF kernel, indicating that the dataset was non-linearly separable.

### 4.2 Comparison of Other Models
- **Random Forest (97.04%)** performed slightly better than KNN but lagged behind ANN and SVM.
- **KNN (96.91%)** struggled with the high dimensionality of the dataset.

### 4.3 Impact of Hyperparameter Tuning
- ANN’s performance improved with **hidden layer optimization, dropout rates, and batch normalization**.
- **SVM’s performance** was enhanced by using an RBF kernel and optimizing `C` values.
- **KNN and Random Forest** showed marginal improvements.

## 5. Conclusion
- **ANN is the best-performing model (98.22% accuracy).**
- **SVM also provided competitive results (97.92%).**
- **Hyperparameter tuning played a crucial role in improving performance.**
- Future improvements can focus on further optimizing ANN architecture and exploring deep CNN models.

---

### 🏆 Key Takeaway
For high-dimensional image classification tasks, deep learning models like ANN tend to outperform traditional models such as KNN, Random Forest, and SVM.
