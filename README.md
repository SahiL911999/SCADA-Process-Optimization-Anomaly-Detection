# SCADA Process Optimization & Anomaly Detection

A comprehensive machine learning solution for industrial process optimization, predictive modeling, and anomaly detection using SCADA (Supervisory Control and Data Acquisition) system data.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![CatBoost](https://img.shields.io/badge/CatBoost-Latest-green.svg)](https://catboost.ai/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Author](#author)
- [Acknowledgments](#acknowledgments)

## 🎯 Overview

This project implements advanced machine learning techniques to analyze industrial SCADA data for process optimization and quality control. The solution addresses real-world manufacturing challenges by:

- **Analyzing correlations** between input parameters and output performance metrics
- **Predicting production outcomes** (Scrap vs. Production mode) with high accuracy
- **Detecting anomalies** in time-series sensor data using deep learning
- **Identifying critical features** that impact process efficiency and product quality

The project was developed as part of the Meta Smart Factory technical assessment, demonstrating expertise in feature engineering, predictive modeling, and industrial data analytics.

## ✨ Features

### 1. **Input-Output Correlation Analysis**
- Comprehensive correlation matrix visualization for 70+ process parameters
- Identification of key relationships between input sensors and output metrics
- Support for dimensionality reduction and feature selection

### 2. **Predictive Modeling with CatBoost**
- Binary classification model for Scrap/Production mode prediction
- **Accuracy: 99.6%** on test data
- Handles categorical features (material types) natively
- Feature importance analysis for interpretability

### 3. **LSTM-Based Anomaly Detection**
- Deep learning autoencoder for time-series anomaly detection
- Sliding window approach (30-step sequences) for temporal pattern recognition
- Identifies top 10 most anomalous features
- Real-time anomaly scoring with configurable thresholds

### 4. **Comprehensive Visualizations**
- Correlation heatmaps for process understanding
- Feature importance rankings
- Anomaly detection heatmaps over time
- Reconstruction error plots with anomaly highlighting

## 📁 Project Structure

```
.
├── code.ipynb                              # Main Jupyter notebook with all analyses
├── filled_dataset_corrected.csv            # Processed dataset (CSV format)
├── filled_dataset_corrected.xlsx           # Processed dataset (Excel format)
├── scrap_mode_catboost_model.cbm          # Trained CatBoost model
├── feature_columns.pkl                     # Saved feature column names
├── correlation_heatmap.png                 # Input-output correlation visualization
├── mean_values_of_output_variables.png    # Output performance analysis
├── input_feature_importance.png            # Feature importance chart
├── anomaly_heatmap.png                     # Per-feature anomaly scores
├── reconstruction_error_anomaly_scores.png # LSTM anomaly detection results
├── ML_Task.pdf                             # Project documentation
├── MSF ML Task (1).pdf                     # Task brief and requirements
├── scada_optimization.pptx (1).pdf        # Presentation slides
├── catboost_info/                          # CatBoost training logs
│   ├── catboost_training.json
│   ├── learn_error.tsv
│   ├── test_error.tsv
│   └── time_left.tsv
└── README.md                               # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/SahiL911999/SCADA-Process-Optimization-Anomaly-Detection.git
cd SCADA-Process-Optimization-Anomaly-Detection
```

2. **Install required packages**
```bash
pip install pandas seaborn matplotlib numpy openpyxl catboost scikit-learn tensorflow keras joblib
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

### Dependencies

```
pandas>=1.3.0
seaborn>=0.11.0
matplotlib>=3.4.0
numpy>=1.21.0
openpyxl>=3.0.0
catboost>=1.0.0
scikit-learn>=1.0.0
tensorflow>=2.6.0
keras>=2.6.0
joblib>=1.0.0
```

## 💻 Usage

### Running the Complete Analysis

Open and run the Jupyter notebook:
```bash
jupyter notebook code.ipynb
```

The notebook is organized into the following sections:

1. **Data Loading & Preprocessing**
2. **Correlation Analysis**
3. **Output Performance Analysis**
4. **CatBoost Classification Model**
5. **LSTM Autoencoder for Anomaly Detection**
6. **Model Saving & Predictions**

### Using the Trained Model

```python
import joblib
from catboost import CatBoostClassifier, Pool

# Load the trained model
model = CatBoostClassifier()
model.load_model('scrap_mode_catboost_model.cbm')

# Load feature columns
feature_columns = joblib.load('feature_columns.pkl')

# Make predictions on new data
# Assuming new_data is a DataFrame with the same features
cat_features = [col for col in feature_columns if 'MATERIAL' in col]
new_pool = Pool(new_data, cat_features=cat_features)
predictions = model.predict(new_pool)
```

## 🔬 Methodology

### 1. Data Preprocessing

- **Dataset**: Industrial SCADA data with 70+ input parameters and 70+ output metrics
- **Cleaning**: Numeric coercion, removal of constant columns, handling missing values
- **Feature Engineering**: Identification of continuous vs. categorical features
- **Scaling**: StandardScaler for LSTM input normalization

### 2. Correlation Analysis

- **Technique**: Pearson correlation coefficient
- **Scope**: Input-input, input-output, and output-output relationships
- **Visualization**: Large-scale heatmap (20x18 inches) for comprehensive analysis

### 3. Predictive Modeling (CatBoost)

**Model Configuration:**
- Iterations: 1000
- Learning Rate: 0.01
- Tree Depth: 6
- Loss Function: Logloss (binary classification)
- L2 Regularization: 3.0
- Train/Test Split: 80/20

**Performance Metrics:**
- Accuracy: **99.6%**
- Confusion Matrix: Near-perfect classification
- Feature Importance: Ranked by contribution to predictions

### 4. Anomaly Detection (LSTM Autoencoder)

**Architecture:**
```
Input (30, n_features)
    ↓
LSTM(128, return_sequences=True)
    ↓
LSTM(64, return_sequences=False)
    ↓
RepeatVector(30)
    ↓
LSTM(64, return_sequences=True)
    ↓
LSTM(128, return_sequences=True)
    ↓
TimeDistributed(Dense(n_features))
```

**Training:**
- Sequence Length: 30 time steps
- Epochs: 30
- Batch Size: 64
- Optimizer: Adam
- Loss: Mean Squared Error (MSE)

**Anomaly Detection:**
- Threshold: 85th percentile of reconstruction errors
- Top 10 anomalous features identified
- Per-feature anomaly scoring over time

## 📊 Results

### Classification Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | 99.6% |
| **Precision (Scrap)** | ~99.5% |
| **Precision (Prod)** | ~99.7% |
| **Recall (Scrap)** | ~99.6% |
| **Recall (Prod)** | ~99.6% |

### Key Insights

1. **Critical Input Features**: The top 10 most important features for predicting production mode have been identified, enabling focused process monitoring.

2. **Output Performance**: Mean values of output variables (columns BG-EP) reveal performance patterns and optimization opportunities.

3. **Anomaly Patterns**: LSTM autoencoder successfully identifies temporal anomalies in sensor data, with clear visualization of anomalous periods.

4. **Correlation Insights**: Strong correlations between specific input-output pairs guide process control strategies.

## 🛠️ Technologies Used

- **Python 3.8+**: Core programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning utilities and preprocessing
- **CatBoost**: Gradient boosting for classification
- **TensorFlow/Keras**: Deep learning framework for LSTM
- **Matplotlib/Seaborn**: Data visualization
- **Jupyter Notebook**: Interactive development environment

## 👨‍💻 Author

**Sahil Ranmbail**

- GitHub: [@sahilranmbail](https://github.com/sahil911999)
- LinkedIn: [Sahil Ranmbail](https://linkedin.com/in/sahil-ranmbail)
- Email: sahil.ranmbail@gmail.com

## 🙏 Acknowledgments

- **Meta Smart Factory** for providing the industrial dataset and technical task brief
- The open-source community for excellent machine learning libraries
- Industrial process optimization research community for methodological guidance

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📈 Future Enhancements

- [ ] Real-time streaming data integration
- [ ] Web dashboard for live monitoring
- [ ] Additional ML models (XGBoost, Random Forest) for comparison
- [ ] SHAP values for enhanced model interpretability
- [ ] Hyperparameter optimization using Optuna
- [ ] Docker containerization for easy deployment
- [ ] REST API for model serving

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/SahiL911999/SCADA-Process-Optimization-Anomaly-Detection/issues).

## 📞 Contact

For questions or collaboration opportunities, please reach out via:
- Email: sahil.ranmbail@gmail.com
- LinkedIn: [Sahil Ranmbail](https://linkedin.com/in/sahil-ranmbail)

---

**Note**: This project demonstrates advanced machine learning techniques for industrial process optimization. The dataset and results are based on real SCADA system data from manufacturing operations.

⭐ If you find this project useful, please consider giving it a star!