Machine Learning || Prediction Model || Logistic Regression || Random Forest || SVM
# Heart Disease Prediction

This repository contains a machine learning project on **Heart Disease Prediction**, where a classification model is built to predict the presence of heart disease based on patient health data.

## Project Overview
Heart disease prediction is a critical healthcare task aimed at early diagnosis to improve patient outcomes. In this project, machine learning models are trained on a dataset containing various health metrics to classify whether a patient is likely to have heart disease.

## Dataset
The dataset used in this project is provided in a CSV file (`heart.csv`). It contains several features representing patient health indicators.

### Features in the Dataset
- **Age**: Age of the patient.
- **Sex**: Gender of the patient (1 = male, 0 = female).
- **CP**: Chest pain type (4 types).
- **Trestbps**: Resting blood pressure (in mm Hg).
- **Chol**: Serum cholesterol (in mg/dl).
- **FBS**: Fasting blood sugar (> 120 mg/dl, 1 = true, 0 = false).
- **Restecg**: Resting electrocardiographic results (values 0, 1, 2).
- **Thalach**: Maximum heart rate achieved.
- **Exang**: Exercise-induced angina (1 = yes, 0 = no).
- **Oldpeak**: ST depression induced by exercise relative to rest.
- **Slope**: The slope of the peak exercise ST segment.
- **Ca**: Number of major vessels (0-3) colored by fluoroscopy.
- **Thal**: A blood disorder indicator (3 = normal, 6 = fixed defect, 7 = reversible defect).
- **Target**: The presence of heart disease (1 = presence, 0 = absence).

## How to Run the Project
1. Clone this repository:
   ```bash
   git clone <repository_url>
   ```
2. Navigate to the project directory:
   ```bash
   cd heart-disease-prediction
   ```
3. Open the Jupyter notebook:
   ```bash
   jupyter notebook Heart_Disease_Prediction.ipynb
   ```
4. Follow the steps in the notebook to preprocess the data, train the model, and evaluate its performance.

## Requirements
- Python 3.8+
- Jupyter Notebook
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## Methodology
1. **Data Preprocessing**:
   - Handling missing values.
   - Encoding categorical features.
   - Feature scaling.

2. **Model Training**:
   - Various classification models are evaluated, including Logistic Regression, Decision Trees, and Random Forest.
   - The best-performing model is selected based on evaluation metrics.

3. **Evaluation**:
   - The model's performance is evaluated using accuracy, precision, recall, and F1-score.

## Results
The notebook provides:
- Confusion matrix and classification report.
- ROC curve and AUC score.
- Insights into feature importance.

## Future Improvements
- Experiment with advanced models such as XGBoost and neural networks.
- Perform hyperparameter tuning to further improve model performance.
- Deploy the model as a web-based diagnostic tool.
