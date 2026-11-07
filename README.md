# Bank Marketing Classification Dashboard

## 1. Problem Statement

The objective of this project is to develop and compare five machine learning classification algorithms to predict whether a customer will subscribe to a term deposit based on bank marketing campaign information.

The project involves dataset preprocessing, model training, performance evaluation and deployment of an interactive Streamlit application.

## 2. Dataset Description

- **Dataset Name:** Bank Marketing Dataset
- **Dataset Source:** UCI Machine Learning Repository
- **Dataset Link:** https://archive.ics.uci.edu/dataset/222/bank+marketing
- **Total Instances:** 45,211
- **Original Input Features:** 16
- **Input Features Used:** 15
- **Target Variable:** `y`
- **Problem Type:** Binary Classification

The target variable represents whether the customer subscribed to a term deposit (`yes` or `no`).

The `duration` feature was excluded because it is only known after the marketing call. Since the prediction objective is to identify potential subscribers before the call, including this feature could introduce target leakage.

## 3. GitHub Repository Link

**GitHub Repository:** https://github.com/sivaperumalashok-star/bank-marketing-ml

## 4. Models Used and Performance Comparison

Five machine learning classification algorithms were implemented and evaluated using the same dataset and test set.

The following six evaluation metrics were calculated for each model:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

### Model Performance Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.901250 | 0.905574 | 0.644483 | 0.347826 | 0.451811 | 0.426058 |
| Decision Tree | 0.874599 | 0.701457 | 0.464880 | 0.475425 | 0.470093 | 0.399021 |
| KNN | 0.896163 | 0.827721 | 0.599002 | 0.340265 | 0.433996 | 0.400128 |
| Naive Bayes | 0.854805 | 0.810095 | 0.405904 | 0.519849 | 0.455864 | 0.377358 |
| Support Vector Machine | 0.905562 | 0.903570 | 0.679577 | 0.364839 | 0.474785 | 0.453129 |

### Observations on Model Performance

**1. Logistic Regression**

Logistic Regression achieved an accuracy of 90.13% and the highest AUC score of 0.905574. This indicates strong class discrimination capability. However, its recall of 0.347826 indicates that it missed a considerable proportion of actual positive subscription cases.

**2. Decision Tree**

The Decision Tree classifier achieved an accuracy of 87.46%. It obtained a recall of 0.475425 and an F1 score of 0.470093. However, its AUC score of 0.701457 was the lowest among the five models, indicating comparatively weaker class discrimination.

**3. K-Nearest Neighbors (KNN)**

KNN achieved an accuracy of 89.62% and an AUC score of 0.827721. Its precision was 0.599002, but its recall was relatively low at 0.340265. This indicates that the model correctly identified a reasonable proportion of predicted positive cases but missed several actual positive cases.

**4. Naive Bayes**

Naive Bayes achieved the highest recall of 0.519849 among all five models. This makes it comparatively better at identifying actual positive subscription cases. However, it recorded the lowest accuracy of 85.48% and precision of 0.405904, indicating a higher number of false positive predictions.

**5. Support Vector Machine (SVM)**

Support Vector Machine achieved the highest accuracy of 90.56%, precision of 0.679577, F1 score of 0.474785 and MCC score of 0.453129. It also achieved the second-highest AUC score of 0.903570. Based on its overall performance across the six evaluation metrics, SVM emerged as the best-performing model.

### Overall Winner for the Dataset

**Support Vector Machine (SVM)** is the overall winner based on the average ranking of all six evaluation metrics.

| Model | Average Rank |
|---|---:|
| Support Vector Machine | 1.500000 |
| Logistic Regression | 2.500000 |
| Decision Tree | 3.500000 |
| KNN | 3.666667 |
| Naive Bayes | 3.833333 |

A lower average rank indicates better overall performance. SVM achieved the lowest average rank of 1.50 and therefore ranked first.

## 5. Streamlit Application Features

The project includes an interactive Streamlit dashboard with the following features:

- Upload test dataset in CSV format.
- Select one of the five trained classification models.
- Display Accuracy, AUC, Precision, Recall, F1 Score and MCC.
- Visualize the confusion matrix.
- Display the classification report.
- Visualize prediction distribution.
- View customer-level predictions.
- Download prediction results as a CSV file.

The application loads the previously trained machine learning pipelines saved using Joblib. It performs predictions and calculates evaluation metrics without retraining the models on the uploaded test dataset.

## 6. Data Preprocessing

The following preprocessing techniques were applied:

- Median imputation for numerical features.
- Most-frequent imputation for categorical features.
- StandardScaler for numerical feature scaling.
- OneHotEncoder for categorical feature encoding.
- Train-test split using an 80:20 ratio.
- Stratified sampling to preserve target class distribution.

## 7. Project Structure

```text
bank-marketing-ml/
├── app.py
├── ML_Assignment.ipynb
├── requirements.txt
├── README.md
├── test_data.csv
├── model_comparison.csv
├── model_ranking.csv
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── svm.joblib
```

## 8. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib
- Matplotlib
- Seaborn

## 9. Installation and Execution

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

Upload the test dataset, select a classification model and view its evaluation results through the interactive dashboard.

## 10. Model Artifacts

All five trained machine learning pipelines are saved in the `model/` directory using Joblib.

Each saved pipeline contains its preprocessing transformations and trained classifier. The Streamlit application loads these saved artifacts to perform predictions without repeating the model training process.
