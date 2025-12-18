# Human Activity Recognition using Smartphone Data

**Team Members:** Ammar Bin Jalal (502254), Abdul Moin (517892)
**Course:** CS 470: Machine Learning

## 1. Abstract
This project implements and compares Classical Machine Learning (SVM, Random Forest) and Deep Learning approaches to classify human activities (Walking, Standing, Laying, etc.) using smartphone sensor data. Our results demonstrate that [SVM] achieved the highest accuracy of [95.42]%, proving effective for high-dimensional sensor classification.

## 2. Introduction
Human Activity Recognition (HAR) is vital for healthcare monitoring and smart environments. The objective of this project is to build robust classifiers that can distinguish between six distinct daily activities using accelerometer and gyroscope data.

## 3. Dataset Description
* **Source:** UCI HAR Dataset
* **Features:** 561 numeric features derived from time and frequency domain sensor signals (tBodyAcc, tGravityAcc, etc.).
* **Target:** 6 Classes (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING).
* **Preprocessing:** Labels were encoded, and features were standardized (Z-score normalization).

## 4. Methodology

### Classical ML Approaches
* **Random Forest:** Implemented with 100 estimators. Used for its robustness to outliers and ability to handle non-linear data.
* **Support Vector Machine (SVM):** Implemented with an RBF kernel (C=10). Chosen for its effectiveness in high-dimensional feature spaces.

### Deep Learning Architecture
* **Model:** Multi-Layer Perceptron (MLP).
* **Structure:** Input -> Dense(64, ReLU) -> Dropout(0.5) -> Dense(32, ReLU) -> Dropout(0.3) -> Output(Softmax).
* **Training:** Adam optimizer, Sparse Categorical Crossentropy loss, Early Stopping to prevent overfitting.

## 5. Results & Analysis

We evaluated three models on the test set. The Support Vector Machine (SVM) achieved the highest performance, demonstrating the effectiveness of kernel-based methods on high-dimensional sensor data.

| Model | Accuracy | F1 Score | Precision |
|-------|----------|----------|-----------|
| Random Forest | 92.6% | 0.925 | 0.927 |
| SVM | 95.4% | 0.954 | 0.954 |
| Deep Learning | 93.2% | 0.932 | 0.936 |

### Key Findings
1.  **Best Model:** SVM achieved the highest accuracy (95.4%).
2.  **Comparison:** Deep Learning was competitive (93.2%), proving that a simple MLP can learn effective representations from raw sensor features. Random Forest lagged slightly (92.6%) likely due to the complexity of the feature space.
3.  **Error Analysis:** The Confusion Matrices revealed that the majority of errors occurred between the **SITTING** and **STANDING** classes. This is expected as these activities have similar accelerometer signatures (static non-vertical). Dynamic activities like **WALKING** were classified with near 100% precision.
## 6. Conclusion
Both classical and deep learning models performed well. While Deep Learning shows great potential, SVM remains a highly efficient and accurate choice for tabular sensor data of this magnitude. Future work could involve using 1D-CNNs for raw signal processing.

## 7. References
1.  Breiman, L. (2001). Random Forests. Machine Learning.
2.  Cortes, C., & Vapnik, V. (1995). Support-vector networks.
3.  Course Project Guidelines.
