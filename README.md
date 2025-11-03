# Employee Attrition Prediction Model

A machine learning project that predicts employee attrition risk using Gradient Boosting Classifier. The model was trained and evaluated on a comprehensive employee dataset, comparing multiple algorithms to identify the best-performing model for deployment.

## 🚀 Live Application

**Streamlit App**: [https://employee-attrition-prediction-model-qx3c2rc4xnikg7hhmxf6za.streamlit.app](https://employee-attrition-prediction-model-qx3c2rc4xnikg7hhmxf6za.streamlit.app)

Try the interactive web application to predict employee attrition risk in real-time!

## 📊 Project Overview

This project addresses the critical challenge of predicting employee attrition in organizations. By analyzing various employee attributes including demographics, job satisfaction, work-life balance, compensation, and career development opportunities, the model helps HR departments proactively identify at-risk employees and implement retention strategies.

### Key Features

- **Real-time Predictions**: Interactive Streamlit application for instant attrition risk assessment
- **Comprehensive Model Comparison**: Tested and evaluated 9 machine learning algorithms across different families
- **Comprehensive Analysis**: 23 employee features analyzed including performance metrics, satisfaction levels, and organizational factors
- **Actionable Insights**: Model provides probability-based risk scores with HR recommendations
- **Production-Ready**: Full pipeline including preprocessing, feature engineering, and model deployment

## 📈 Model Performance

### Models Tested

We evaluated **9 different machine learning algorithms** across various model families:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Training Time (sec) | Test Time (sec) | Interpretability |
|-------|----------|-----------|--------|----------|---------|---------------------|-----------------|------------------|
| **Gradient Boosting** ✅ | **0.7613** | **0.7521** | **0.7429** | **0.7475** | **0.8497** | 21.54 | 0.11 | Medium |
| LightGBM | 0.7584 | 0.7471 | 0.7436 | 0.7454 | 0.8472 | 2.24 | 0.57 | Medium |
| XGBoost | 0.7578 | 0.7475 | 0.7410 | 0.7442 | 0.8464 | 1.73 | 0.10 | Medium |
| Random Forest | 0.7494 | 0.7384 | 0.7325 | 0.7355 | 0.8373 | 15.44 | 2.03 | Medium |
| SVM | 0.7469 | 0.7356 | 0.7302 | 0.7329 | 0.8333 | 385.99 | 32.38 | High |
| Logistic Regression | 0.7476 | 0.7376 | 0.7281 | 0.7328 | 0.8336 | 0.78 | 0.01 | High |
| MLP Classifier | 0.7435 | 0.7290 | 0.7329 | 0.7310 | 0.8302 | 86.21 | 0.05 | Low |
| Decision Tree | 0.7155 | 0.7131 | 0.6722 | 0.6920 | 0.7990 | 0.18 | 0.02 | High |
| KNN | 0.6592 | 0.6366 | 0.6602 | 0.6482 | 0.7069 | 0.28 | 10.29 | Low |

**Selected Model**: Gradient Boosting Classifier was chosen as the best-performing model based on comprehensive evaluation across all metrics.

### Model Families Tested

1. **Ensemble Methods** (4): Gradient Boosting, LightGBM, XGBoost, Random Forest
2. **Linear Models** (2): Logistic Regression, SVM
3. **Tree-based** (1): Decision Tree
4. **Instance-based** (1): KNN
5. **Neural Networks** (1): MLP Classifier

### Why Gradient Boosting?

Among all 9 models tested, the Gradient Boosting Classifier achieved:
- **Highest Accuracy** (76.13%)
- **Best ROC-AUC Score** (0.8497) - superior class separation
- **Superior F1-Score** (0.7475) - best balance of precision and recall
- **Highest Precision** (75.21%) - minimizing false positives
- **Fast Prediction** (0.11 sec) - suitable for real-time deployment
- **Good Interpretability** - feature importance plots available
- **Reasonable Training Time** - 21.54 seconds

**Trade-offs Considered:**
- LightGBM and XGBoost had faster training (1.73-2.24 sec) but lower accuracy
- SVM had very slow training (386 sec) with lower performance
- Logistic Regression had fast training (0.78 sec) but lower accuracy
- Decision Tree overfitted with lower generalization

## 📁 Project Structure

```
Employee-Attrition-Prediction-Model/
│
├── app.py                          # Streamlit deployment application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
│
├── dataset_attrition/              # Raw datasets
│   ├── train.csv                   # Training set (59,598 samples)
│   └── test.csv                    # Test set (14,900 samples)
│
├── splits/                         # Preprocessed data splits
│   ├── X_train.csv                 # Training features
│   ├── X_val.csv                   # Validation features
│   ├── X_test.csv                  # Test features
│   ├── y_train.csv                 # Training labels
│   └── y_val.csv                   # Validation labels
│
├── models/                         # Trained models and results
│   ├── gradiant_boosting/          # Selected best model
│   │   ├── gbm_pipeline.pkl        # Complete preprocessing + model pipeline
│   │   ├── gbm_model.pkl           # Trained model
│   │   ├── gbm_columns.json        # Feature column order
│   │   ├── gbm_results.csv         # Performance metrics
│   │   ├── gbm_meta.json           # Training metadata
│   │   ├── gbm_feature_importance.png
│   │   └── *.ipynb                 # Model notebooks
│   │
│   ├── lightgbm/                   # LightGBM model artifacts
│   │   ├── lgbm_model.pkl
│   │   ├── lgbm_results.csv
│   │   └── *.ipynb
│   │
│   └── xgboost/                    # XGBoost model artifacts
│       ├── xgb_model.pkl
│       ├── xgb_results.csv
│       └── *.ipynb
│
├── preprocessing.ipynb             # Data preprocessing pipeline
│
├── gbm_pipeline.pkl                # Production pipeline (deployed)
└── gbm_columns.json                # Production feature columns
```

## 🔧 Technical Stack

- **Language**: Python 3
- **ML Framework**: Scikit-learn
- **Models**: Gradient Boosting, LightGBM, XGBoost, Random Forest, SVM, Logistic Regression, MLP, Decision Tree, KNN
- **Deployment**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Feature importance plots

## 📊 Dataset Information

### Dataset Size
- **Training Set**: 59,598 employees
- **Test Set**: 14,900 employees
- **Target Distribution**: Well-balanced (52.5% Stayed, 47.5% Left)

### Features (23 Total)

**Demographic Features:**
- Age, Gender
- Marital Status
- Number of Dependents

**Employment Features:**
- Years at Company
- Company Tenure
- Job Role (5 categories: Finance, Healthcare, Technology, Education, Media)
- Job Level (Entry, Mid, Senior)
- Monthly Income
- Number of Promotions

**Work Environment:**
- Distance from Home
- Remote Work
- Overtime
- Company Size

**Performance & Satisfaction (Ordinal):**
- Work-Life Balance (Poor → Excellent)
- Job Satisfaction (Very Low → High)
- Performance Rating (Low → High)
- Company Reputation (Very Poor → Excellent)
- Employee Recognition (Very Low → High)

**Opportunities:**
- Leadership Opportunities
- Innovation Opportunities

## 🔄 Data Preprocessing Pipeline

### 1. **Data Cleaning**
- Removed identifier columns (Employee ID)
- Created backup copies of raw data

### 2. **Missing Value Handling**
- Dataset had no missing values
- Implemented handlers for future cases (median for numeric, mode for categorical)

### 3. **Outlier Treatment**
- Applied Winsorization (1st-99th percentile capping)
- Features capped: Age, Years at Company, Monthly Income, Number of Promotions, Distance from Home, Company Tenure

### 4. **Feature Encoding**
- **Ordinal Encoding**: 5 features with natural order (Work-Life Balance, Job Satisfaction, Performance Rating, Company Reputation, Employee Recognition)
- **One-Hot Encoding**: 9 categorical features (Gender, Job Role, Education Level, Marital Status, Job Level, Company Size, Remote Work, Leadership Opportunities, Innovation Opportunities)

### 5. **Scaling**
- **StandardScaler**: Applied to all numeric features (mean=0, std=1)

### 6. **Data Splitting**
- Train/Validation split: 80/20 with stratification
- Final feature count: 31 after encoding

## 🤖 Model Architecture

### Gradient Boosting Classifier

```python
GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
```

**Pipeline Components:**
1. **Winsorizer**: Custom transformer for outlier capping
2. **ColumnTransformer**: Preprocessing (StandardScaler)
3. **GradientBoostingClassifier**: Final model

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip or conda package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/Employee-Attrition-Prediction-Model.git
cd Employee-Attrition-Prediction-Model
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the Streamlit app**
```bash
streamlit run app.py
```

The application will launch at `http://localhost:8501`

## 💻 Usage

### Streamlit Application

1. Open the application in your browser
2. Enter employee information across categories:
   - **Basic Info**: Age, Gender, Job Role, etc.
   - **Compensation**: Monthly Income, Job Level
   - **Work Environment**: Remote Work, Overtime, Distance from Home
   - **Performance Metrics**: Job Satisfaction, Work-Life Balance, Performance Rating
   - **Company Factors**: Company Size, Company Reputation
   - **Opportunities**: Leadership and Innovation opportunities

3. Adjust prediction threshold if needed (default: 0.5)

4. Click "Predict risk" to get:
   - Probability of attrition
   - Risk classification (High/Low)
   - Actionable HR recommendations
   - Interpretive insights

### Using the Model Programmatically

```python
import joblib
import pandas as pd
import json

# Load pipeline and columns
pipe = joblib.load("gbm_pipeline.pkl")
with open("gbm_columns.json", "r") as f:
    train_columns = json.load(f)

# Prepare employee data
employee_data = pd.DataFrame([{
    "Age": 35,
    "Years at Company": 5,
    "Monthly Income": 6000,
    # ... other features
}])

# Predict
probability = pipe.predict_proba(employee_data)[:, 1][0]
prediction = "High Risk" if probability >= 0.5 else "Low Risk"
```

## 📋 Model Evaluation

### Performance Metrics Explained

- **Accuracy**: Overall correctness (76.13%)
- **Precision**: Correctly identified at-risk employees (75.21%)
- **Recall**: Proportion of actual leavers identified (74.29%)
- **F1-Score**: Harmonic mean of precision and recall (74.75%)
- **ROC-AUC**: Model's ability to distinguish classes (0.8497)

### Feature Importance

The model prioritizes factors such as:
- Job Satisfaction
- Work-Life Balance
- Monthly Income
- Years at Company
- Performance Rating
- Employee Recognition

*(Check `gbm_feature_importance.png` for detailed visualizations)*

## 🎯 Business Impact

### Use Cases

1. **Proactive Retention**: Identify at-risk employees before they decide to leave
2. **Resource Allocation**: Focus retention efforts on high-risk employees
3. **Cost Reduction**: Reduce recruitment and training costs
4. **Data-Driven HR**: Make evidence-based decisions about employee retention
5. **Employee Satisfaction**: Improve overall workplace conditions

### Recommendations Generated

The application provides tailored recommendations based on risk factors:
- Salary adjustments for underpaid employees
- Work-life balance improvements
- Career development opportunities
- Performance improvement plans
- Recognition and engagement initiatives

## 🔬 Model Comparison Methodology

All models were evaluated using:
- **Same train/validation splits** (stratified)
- **Consistent preprocessing pipeline**
- **Multiple evaluation metrics**
- **Feature importance analysis**

Decision criteria prioritized model performance over training speed.

## 🛠️ Development Workflow

1. **Data Preprocessing** (`preprocessing.ipynb`)
   - Data cleaning, encoding, scaling
   - Train/val/test splits

2. **Model Training** (`*_train.ipynb`)
   - Hyperparameter tuning
   - Cross-validation
   - Model persistence

3. **Model Evaluation** (`*_evaluate.ipynb`)
   - Performance metrics
   - Feature importance
   - Confusion matrices

4. **Pipeline Building** (`build_gbm_pipeline.ipynb`)
   - End-to-end pipeline
   - Deployment preparation

5. **Deployment** (`app.py`)
   - Streamlit interface
   - Real-time predictions

## 📝 Requirements

```
streamlit==1.39
pandas
joblib
scikit-learn==1.6.1
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Authors

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- Project Link: [https://github.com/yourusername/Employee-Attrition-Prediction-Model](https://github.com/yourusername/Employee-Attrition-Prediction-Model)

## 🙏 Acknowledgments

- Dataset sources and contributors
- Scikit-learn team for excellent ML tools
- Streamlit for easy ML deployment
- The open-source community

## 📞 Contact

For questions, suggestions, or collaborations, please open an issue on GitHub.

---

⭐ **Star this repo if you find it helpful!**
