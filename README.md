# 🏥 Haberman Survival Dataset - Machine Learning Classification 

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Complete-success)](https://github.com/TasinAhmed2508/ML-Haberman-Dataset)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 📋 Overview

This project applies **multiple machine learning algorithms** to predict patient survival outcomes using the **Haberman Survival Dataset**. The goal is to classify whether a patient survived 5 years or more after undergoing breast cancer surgery based on clinical features.

The project implements and compares **7 different classification models** to identify the most effective algorithm for this binary classification task.

## 📁 Folder Structure

```
ML-Haberman-Dataset/
│
├── 📓 notebooks/
│   ├── haberman_knn.ipynb                 # K-Nearest Neighbors classifier
│   ├── haberman_logistic.ipynb            # Logistic Regression
│   ├── haberman_naive_bayes.ipynb         # Gaussian Naive Bayes
│   ├── haberman_decision_tree.ipynb       # Decision Tree Classifier
│   ├── haberman_random_forest.ipynb       # Random Forest Classifier
│   ├── haberman_SVM.ipynb                 # Support Vector Machine
│   └── haberman_boosting.ipynb            # Gradient Boosting Classifier
│
├── 📊 haberman.csv                        # Dataset file
├── 📄 README.md                           # Project documentation
├── 📜 LICENSE                             # MIT License
├── 📋 requirements.txt                    # Python dependencies
└── 🔒 .gitignore                          # Git ignore rules
```

## 📊 Dataset

**Name:** Haberman's Survival Dataset  
**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/haberman's+survival)  
**Size:** 306 samples × 4 features  
**Target Variable:** `Survival Status` (1 = Survived 5+ years, 2 = Did not survive 5+ years)

### Features:
- **Age**: Patient's age at surgery
- **Year**: Year of surgery
- **Positive Nodes**: Number of positive axillary lymph nodes detected
- **Survival Status**: Target variable (binary classification)

## 🎯 Key Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Random Forest** | 78.4% | 0.82 | 0.75 | 0.78 |
| **Gradient Boosting** | 77.1% | 0.81 | 0.73 | 0.77 |
| **SVM** | 76.5% | 0.79 | 0.72 | 0.75 |
| **Decision Tree** | 75.3% | 0.78 | 0.70 | 0.74 |
| **Logistic Regression** | 74.8% | 0.76 | 0.69 | 0.72 |
| **Naive Bayes** | 73.5% | 0.74 | 0.68 | 0.71 |
| **KNN (k=5)** | 72.9% | 0.72 | 0.67 | 0.69 |

**Best Performer:** Random Forest Classifier with **78.4% accuracy**

## 🚀 Installation & Usage

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/TasinAhmed2508/ML-Haberman-Dataset.git
   cd ML-Haberman-Dataset
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebooks:**
   - Navigate to the `notebooks/` folder
   - Open any `.ipynb` file with Jupyter Notebook:
     ```bash
     jupyter notebook notebooks/
     ```
   - Select a notebook to run (e.g., `haberman_random_forest.ipynb`)

### Individual Model Execution
   ```bash
   jupyter notebook notebooks/haberman_random_forest.ipynb
   ```

## 📚 Algorithms Implemented

1. **K-Nearest Neighbors (KNN)** - Simple instance-based learning
2. **Logistic Regression** - Linear classification with probability estimates
3. **Naive Bayes** - Probabilistic classifier based on Bayes' theorem
4. **Decision Tree** - Tree-based hierarchical classifier
5. **Random Forest** - Ensemble of decision trees (BEST MODEL)
6. **Support Vector Machine (SVM)** - Maximum margin classifier
7. **Gradient Boosting** - Sequential ensemble learning

## 🔧 Technologies Used

- **Python 3.8+**
- **scikit-learn** - Machine learning algorithms
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computations
- **matplotlib** - Data visualization
- **seaborn** - Statistical data visualization
- **Jupyter Notebook** - Interactive analysis environment

## 📈 Key Findings

- **Feature Importance**: Number of positive nodes is the most predictive feature
- **Model Comparison**: Ensemble methods (Random Forest, Boosting) outperform individual classifiers
- **Cross-validation**: Models were validated using 5-fold cross-validation
- **Scalability**: Data preprocessing includes normalization for distance-based algorithms

## 👨‍💻 Author

**Tasin Ahmed**

### Connect with Me:
- 🐙 [GitHub Profile](https://github.com/TasinAhmed2508)
- 📧 Open to collaborations and discussions on ML projects

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

**Copyright © 2026 Tasin Ahmed**

---

**Last Updated:** January 2026  
**Status:** ✅ Complete and Ready for Production
