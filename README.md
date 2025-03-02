# Intro-to-Machine-Learning

## Project Overview
This project focuses on predicting the survival status of patients with cirrhosis using machine learning techniques. The dataset used originates from the University of California, Irvine (UCI), containing clinical characteristics of patients. The goal is to develop predictive models using statistical and machine learning techniques.

## Features
- **Data Preprocessing**:
  - Handling missing values
  - Feature scaling and encoding
  - Dimensionality reduction with PCA
- **Machine Learning Models**:
  - K-Nearest Neighbors (KNN)
  - Support Vector Machine (SVM)
  - Decision Tree (DT)
- **Model Evaluation**:
  - Accuracy, Precision, Recall, and F1-score
  - Balanced accuracy for handling imbalanced data
- **Model Selection**:
  - Comparison of different models
  - Hyperparameter tuning using GridSearchCV

## Installation
1. Clone the repository:
   ```sh
   git clone <repository_url>
   cd ml-introduction-project
   ```
2. Install dependencies:
   ```sh
   pip install numpy pandas scikit-learn matplotlib seaborn
   ```
   
## Dataset
- **Source**: UCI Cirrhosis Patient Survival Dataset
- **Size**: 418 instances, 20 variables
- **Target Variable**: `Status`
  - 0: Death
  - 1: Censored
  - 2: Censored due to liver transplantation

## Model Development Process
1. **Data Analysis and Preprocessing**:
   - Recoding categorical variables
   - Handling missing values using imputation
   - Standardizing numerical features
   - Removing redundant/noisy features
   - Applying Principal Component Analysis (PCA) for dimensionality reduction
2. **Model Training and Evaluation**:
   - Implementing KNN, SVM, and Decision Tree classifiers
   - Evaluating models with cross-validation
   - Addressing class imbalance using class weighting
3. **Model Selection**:
   - Comparing performance across models
   - Selecting the best-performing model based on evaluation metrics

## Results
- The **SVM model** with class weighting and outlier removal performed best.
- Achieved a balance between underfitting and overfitting.
- Final evaluation showed improved accuracy and F1-score on test data.

## Future Improvements
- Explore ensemble methods like Random Forest or Gradient Boosting.
- Implement additional feature engineering techniques.
- Optimize hyperparameters using Bayesian optimization.

## References
- UCI Machine Learning Repository: Cirrhosis Patient Dataset

## Author
- Zhihao Chen

