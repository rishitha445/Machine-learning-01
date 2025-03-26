# 💳 Loan Default Prediction with LightGBM  

## 📌 **Project Overview**  
This project demonstrates how to predict loan defaults using **LightGBM**—a highly efficient gradient boosting framework. Financial institutions face significant risks from loan defaults, making accurate predictions critical for reducing losses and improving lending strategies.  

In this project, we develop a LightGBM model to predict loan defaults, focusing on handling class imbalance and enhancing model interpretability through SHAP analysis and advanced visualizations.  

---  

## 🎯 **Objectives**  
✅ Develop a predictive model using LightGBM to classify loan defaults.  
✅ Optimize model performance through hyperparameter tuning using **GridSearchCV**.  
✅ Address class imbalance using techniques like threshold adjustments and weighting.  
✅ Evaluate model performance using ROC-AUC, classification reports, and confusion matrices.  
✅ Provide interpretability using SHAP values and feature importance.  

---

## 📂 **Dataset Overview**  
- **Dataset:** Loan Default Dataset  
- **Training Samples:** 204,277  
- **Test Samples:** 51,070  
- **Total Features:** 18 (9 numerical + 7 categorical + 1 target)  

### **Feature Categories**  
| Feature Type | Examples |
|-------------|----------|
| **Numerical** | Age, Income, LoanAmount, CreditScore, InterestRate, MonthsEmployed |
| **Categorical** | Education, EmploymentType, MaritalStatus, HasMortgage, LoanPurpose |
| **Target** | Default (0 = No, 1 = Yes) |  

---

## 🏗️ **Model: LightGBM Classifier**  
LightGBM (Light Gradient Boosting Machine) is a decision-tree-based algorithm that trains models using histogram-based techniques, making it highly efficient for large datasets.  

### 🚀 **Why LightGBM?**  
✅ Faster training using histogram-based algorithms.  
✅ Handles categorical features directly without one-hot encoding.  
✅ Supports parallel and GPU-based processing.  
✅ Excellent handling of imbalanced data through cost-sensitive learning.  

---

## 🔧 **Model Training**  
### **Hyperparameter Tuning**  
We performed hyperparameter tuning using **GridSearchCV** with the following search space:  

| Parameter | Value Range | Purpose |
|-----------|-------------|---------|
| `num_leaves` | 31, 50 | Complexity control |
| `max_depth` | -1, 10, 20 | Depth of the trees |
| `learning_rate` | 0.05, 0.1 | Step size for boosting |
| `n_estimators` | 100, 150 | Number of boosting iterations |


## ⚙️ **How to Run the Code**  
### 1. **Clone the Repository**  
```bash
git clone https://github.com/yourusername/loan-default-lgbm.git




