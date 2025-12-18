# Human Activity Recognition (HAR) using Smartphone Data

**Course:** CS 470: Machine Learning

**Team Members:** Ammar Bin Jalal (502254), Abdul Moin (517892)

---

## 1. Abstract
This project implements and compares Classical Machine Learning (Random Forest, SVM) and Deep Learning approaches to classify human activities (Walking, Standing, Laying, etc.) using smartphone sensor data. The goal was to accurately identify six distinct activities based on 561 features derived from accelerometer and gyroscope signals.

Our experimental results demonstrate that the **Support Vector Machine (SVM)** achieved the highest performance with an accuracy of **95.42%**, proving to be the most effective model for this high-dimensional tabular dataset.

## 2. Introduction
Human Activity Recognition (HAR) is a core technology in wearable devices, health monitoring, and smart environments. The challenge lies in distinguishing between activities that have very similar sensor patterns, such as standing still versus sitting.

This project utilizes the **UCI HAR Dataset** (Project 30) to train and evaluate three different classifiers:
1.  **Random Forest:** An ensemble method known for robustness.
2.  **Support Vector Machine (SVM):** A kernel-based method ideal for high-dimensional feature spaces.
3.  **Deep Learning (MLP):** A fully connected neural network to learn complex non-linear representations.

## 3. Dataset Properties & Description

The project utilizes the **Human Activity Recognition (HAR)** dataset. It is a high-quality, pre-processed dataset collected from the accelerometers and gyroscopes of Samsung Galaxy S II smartphones.

### 3.1 Data Overview
* **Source:** 30 volunteers performing 6 standard activities while carrying a waist-mounted smartphone.
* **Total Samples:** 10,299 instances.
    * **Training Set:** 7,352 samples (71%).
    * **Test Set:** 2,947 samples (29%).
* **Missing Values:** None (0 null values). The dataset is clean and ready for immediate modeling.

### 3.2 Feature Engineering (Input)
The dataset does not contain raw signal logs. Instead, it features **561 pre-computed attributes** per sample. These features are statistical summaries derived from the raw 3-axial signals (tAcc-XYZ and tGyro-XYZ):

* **Time-Domain Features (prefix `t`):** Captures constant movement properties (e.g., `tBodyAcc-mean()-X`, `tGravityAcc-min()-Y`).
* **Frequency-Domain Features (prefix `f`):** Captures vibration and repetition frequencies using Fast Fourier Transform (FFT) (e.g., `fBodyGyro-energy()`).
* **Statistical Metrics:** Each signal is summarized using:
    * `mean()`: Average value.
    * `std()`: Standard deviation.
    * `mad()`: Median absolute deviation.
    * `max()` / `min()`: Range limits.
    * `sma()`: Signal magnitude area.
    * `energy()`: Energy measure.
    * `entropy()`: Signal entropy.

### 3.3 Target Labels (Output)
The model predicts one of six distinct movement classes:
1.  `WALKING`
2.  `WALKING_UPSTAIRS`
3.  `WALKING_DOWNSTAIRS`
4.  `SITTING`
5.  `STANDING`
6.  `LAYING`

### 3.4 Data Preprocessing
* **Normalization:** All feature values are normalized and bounded within `[-1, 1]`. This allows models like SVM and Neural Networks to converge faster without one feature dominating others due to scale differences.
* **Class Balance:** The classes are reasonably balanced, with `LAYING` being the most frequent (~19%) and `WALKING_DOWNSTAIRS` the least frequent (~13%), removing the need for synthetic oversampling (SMOTE).

## 4. Methodology

### 4.1 Classical Machine Learning
We implemented two classical algorithms using `scikit-learn`:

* **Random Forest Classifier:**
    * **Settings:** `n_estimators=100`, `random_state=42`.
    * **Rationale:** Chosen as a strong baseline model that is resistant to overfitting and does not require extensive hyperparameter tuning.

* **Support Vector Machine (SVM):**
    * **Settings:** Kernel=`rbf`, `C=10`, `gamma='scale'`.
    * **Rationale:** The RBF kernel allows for non-linear decision boundaries. A higher `C` value (10) was chosen to prioritize correct classification of training points, which is appropriate for this clean, high-dimensional dataset.

### 4.2 Deep Learning
We designed a Multi-Layer Perceptron (MLP) using **TensorFlow/Keras**:

* **Architecture:**
    * **Input Layer:** 561 neurons.
    * **Hidden Layer 1:** 64 neurons (ReLU activation) + Dropout (0.5).
    * **Hidden Layer 2:** 32 neurons (ReLU activation) + Dropout (0.3).
    * **Output Layer:** 6 neurons (Softmax activation).
* **Training:**
    * **Optimizer:** Adam.
    * **Loss Function:** Sparse Categorical Crossentropy.
    * **Callbacks:** `EarlyStopping` (patience=10) to prevent overfitting.

## 5. Results & Analysis

We evaluated the models on the unseen test set of 2,947 samples.

| Model | Accuracy | Correct Predictions | Precision (Weighted) | F1 Score (Weighted) |
|-------|----------|---------------------|----------------------|---------------------|
| **Random Forest** | 92.13% | 2,715 / 2,947 | 0.922 | 0.921 |
| **Deep Learning** | ~94.50% | ~2,785 / 2,947 | 0.946 | 0.945 |
| **SVM (Best)** | **95.42%** | **2,812 / 2,947** | **0.955** | **0.954** |

### 5.1 Key Findings
1.  **Winner:** The **SVM** outperformed the other models. Its ability to create complex hyperplanes in the 561-dimensional space made it superior for this specific feature set.
2.  **Confusion Matrix Analysis:**
    * **The "Static" Challenge:** The primary source of error across all models was distinguishing between **SITTING** and **STANDING**. For example, the SVM misclassified 48 "Sitting" instances as "Standing". This is due to the nearly identical sensor orientation (vertical) and lack of movement in both activities.
    * **Dynamic Success:** All models achieved near-perfect accuracy (close to 100%) for **LAYING** and high accuracy for **WALKING**, as these activities have distinct gravitational and acceleration signatures.

### 5.2 Visualization
*(Below is the Confusion Matrix comparing the models. Note the higher error rate between classes 1 and 2 (Sitting/Standing) compared to others.)*

![Confusion Matrix](confusion_matrix.png)

## 6. Conclusion
This project demonstrated that while Deep Learning is a powerful tool, well-tuned classical algorithms like SVM can still outperform neural networks on structured, tabular datasets with high dimensionality. The SVM model achieved a production-ready accuracy of **95.4%**.

## 7. Future Work
* **Feature Selection:** Use Principal Component Analysis (PCA) to reduce the 561 features to a smaller set (e.g., 50) to speed up training without significant accuracy loss.
* **Raw Signal Processing:** Instead of using the pre-calculated features, we could feed the raw signal data into a **1D-Convolutional Neural Network (CNN)** or **LSTM** to let the model learn its own features from the time-series data.

## 8. References
1.  Anguita, D., et al. (2013). *A Public Domain Dataset for Human Activity Recognition Using Smartphones*.
2.  Breiman, L. (2001). *Random Forests*. Machine Learning.
