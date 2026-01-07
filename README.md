# Diabetes Prediction & ML Classifiers Comparison

This project focuses on predicting diabetes using the **Pima Indians Diabetes Dataset**. It involves data preprocessing, exploratory data analysis (EDA), and a detailed comparison of multiple Machine Learning models to identify the most effective approach for classification.

##  Overview
The goal of this assignment was to implement and evaluate various classification algorithms and analyze the impact of feature selection and hyperparameter tuning on model performance.

##  Models Implemented
The following algorithms were implemented and compared:
* **Logistic Regression**
* **Linear Discriminant Analysis (LDA)**
* **Random Forest**
* **Support Vector Machines (SVM)** - with Linear & RBF kernels
* **K-Nearest Neighbors (KNN)**
* **Naive Bayes**
* **Ensemble Learning** (Bagging & Voting Classifiers)

##  Key Features
- **Data Preprocessing:** Handling missing values, scaling features using `StandardScaler`.
- **Feature Selection:** Comparison between using all 8 features vs. the top 5 most relevant features.
- **Model Tuning:** Experimenting with SVM parameters ($C=1, C=100$) and different kernels.
- **Evaluation Metrics:** Detailed analysis using Confusion Matrix, Accuracy, Precision, Recall, and F1-Score.
- **Visualizations:** Correlation heatmaps, accuracy comparison plots, and ROC curves.

##  Project Structure
* `HW2_ML.ipynb`: The main Jupyter Notebook containing code, visualizations, and analysis.
* `ml2zandi.py`: Python script version of the implementation.
* `diabetes.csv`: The dataset used for training and testing.

##  Results Summary
- The impact of removing less significant features (like SkinThickness) was analyzed, showing that model complexity can be reduced without significant loss in accuracy.
- SVM performance was evaluated under different kernels, highlighting the non-linear nature of the data.
- Ensemble methods (Bagging/Voting) were used to improve the stability and performance of the base learners.

##  How to Use
1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/diabetes-classification-ml.git](https://github.com/your-username/diabetes-classification-ml.git)
