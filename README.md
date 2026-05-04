#                                   🏋️ Body Performance Intelligence System (Hayper Digi)
<p align="center">
  <img src="images/banner.png" alt="Body Performance Analytics Banner" width="100%">
</p>

#                                    🏋️ Body Performance Intelligence System (Hayper Digi)

![Status](https://img.shields.io/badge/Status-Completed-success.svg)
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)

An end-to-end Machine Learning pipeline to classify physical performance levels based on biometric data.

## 👥 Team Members (Hayper Digi)
- **Mohamed Khaled Mahmoud** (Team Leader)
- **Mohamed Eid Abdelkhalek**
- **Youssef Mohamed Elkoumy**
- **Moamen Essam Omar**
- **Mahmoud Maher**

---

## 📊 1. Exploratory Data Analysis (EDA)
We conducted a comprehensive EDA to uncover data distributions, anomalies, and relationships.

### Data Distributions & Outliers
<p align="center">
  <img src="images/histogram_distributions.png" alt="Histograms" width="48%">
  <img src="images/categorical_distributions.png" alt="Categorical" width="48%">
</p>
<p align="center">
  <img src="images/boxplot_outliers.png" alt="Boxplots" width="80%">
</p>

### Feature Relationships & Importance
<p align="center">
  <img src="images/correlation_heatmap.png" alt="Correlation Heatmap" width="48%">
  <img src="images/scatter_plots.jpg" alt="Scatter Plots" width="48%">
</p>
<p align="center">
  <img src="images/feature_importance.png" alt="Feature Importance" width="80%">
</p>

---

## ⚙️ 2. Hyperparameter Tuning
Optimizing our models to ensure the highest possible accuracy without overfitting.
<p align="center">
  <img src="images/hyperparameter_tuning.png" alt="Hyperparameter Tuning" width="80%">
</p>

---

## 🏆 3. Classification Results
Evaluating our Machine Learning models on their ability to correctly classify physical performance into 4 distinct grades (A, B, C, D).

<p align="center">
  <img src="images/classification_comparison.png" alt="Classification Comparison" width="48%">
  <img src="images/best_model_comparison_chart.jpg" alt="Best Model Chart" width="48%">
</p>

### Confusion Matrices & Split Stability
Analyzing where our models succeed, where adjacent physiological classes overlap, and ensuring model stability across different Train/Test splits.
<p align="center">
  <img src="images/confusion_matrices.png" alt="Confusion Matrices" width="48%">
  <img src="images/split_comparison.png" alt="Train/Test Split Comparison" width="48%">
</p>

---

## 📈 4. Regression Analysis
In addition to classification, we built regression models to predict exact physical output metrics (e.g., Broad Jump distance).
<p align="center">
  <img src="images/regression_comparison.png" alt="Regression Models Comparison" width="48%">
  <img src="images/best_regressor_analysis.jpg" alt="Best Regressor Analysis" width="48%">
</p>

---

## 📁 Repository Structure
```text
├── data/
│   └── bodyPerformance.csv
├── images/
│   ├── banner.png
│   ├── best_model_comparison_chart.jpg
│   ├── best_regressor_analysis.jpg
│   ├── boxplot_outliers.png
│   ├── categorical_distributions.png
│   ├── classification_comparison.png
│   ├── confusion_matrices.png
│   ├── correlation_heatmap.png
│   ├── feature_importance.png
│   ├── histogram_distributions.png
│   ├── hyperparameter_tuning.png
│   ├── regression_comparison.png
│   ├── scatter_plots.jpg
│   └── split_comparison.png
├── main.py
├── Body_Performance_Report.pdf
└── README.md