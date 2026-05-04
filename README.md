#                                   🏋️ Body Performance Intelligence System (Hayper Digi)
<p align="center">
  <img src="images/banner.png" alt="Body Performance Analytics Banner" width="100%">
</p>

# 🏋️ Body Performance Analytics & Intelligent Classification System
### Introduction to AI & Machine Learning Project | Team Hayper Digi

---

## 📝 1. Project Objective
The primary goal of this project is to develop a robust Machine Learning pipeline capable of:
1. **Classification:** Categorizing individuals into four physical performance grades (**A, B, C, D**) based on biometric and physical test data.
2. **Regression:** Predicting the **Broad Jump** distance (power metric) based on other physiological attributes.

## 👥 2. Team Members (Hayper Digi)
| Role | Name |
| :--- | :--- |
| **👑 Team Leader** | Mohamed Khaled Mahmoud |
| **Member** | Mohamed Eid Abdelkhalek |
| **Member** | Youssef Mohamed Elkoumy |
| **Member** | Moamen Essam Omar |
| **Member** | Mahmoud Maher |

---

## 📂 3. Dataset Overview
We utilized the **Body Performance Dataset**, which contains **13,393 records**.
- **Biometric Data:** Age, Gender, Height, Weight, Body Fat %, Diastolic/Systolic BP.
- **Performance Metrics:** Grip Force, Sit and Bend Forward, Sit-ups, Broad Jump.
- **Target:** Performance Class (A=Best, D=Lowest).

---

## ⚙️ 4. Project Workflow (Step-by-Step)

### Phase 1: Data Preprocessing & Cleaning
In this initial stage, we ensured data integrity through:
- **Standardization:** Renaming columns and removing special characters.
- **Integrity Checks:** Identifying and handling missing values and duplicate records.
- **Outlier Management:** Using the Interquartile Range (IQR) method to cap extreme physiological values (e.g., Body Fat % and Blood Pressure) to prevent model distortion.

### Phase 2: Exploratory Data Analysis (EDA)
Instead of just viewing plots, we analyzed underlying patterns:
- **Correlation Mapping:** Identifying how `sit-ups` and `grip force` strongly influence the final performance grade.
- **Distribution Analysis:** Examining the balance between genders and age groups to ensure a non-biased model.

### Phase 3: Feature Engineering & Selection
We boosted model performance by creating 7 new features, including:
- **BMI (Body Mass Index):** A critical indicator of physical health.
- **Fitness Score:** A composite metric derived from multiple physical tests.
- **Feature Selection:** Using a 3-method ensemble (SelectKBest, Random Forest Importance, and Correlation) to select the top 10 most impactful features.

### Phase 4: Model Implementation
We implemented a multi-model architecture to compare performance:
- **Classification:** KNN, Decision Trees, SVM (Linear & RBF), Neural Networks (MLP), and Hist-Gradient Boosting.
- **Regression:** Linear Regression, Ridge, and Gradient Boosting Regressor.
- **Hyperparameter Tuning:** Using GridSearchCV to find the optimal parameters for each model.

### Phase 5: Evaluation & Testing
Models were evaluated using:
- **Accuracy & F1-Score** for classification.
- **R² & RMSE** for regression.
- **Cross-Validation:** Ensuring stability across different data splits (80/20, 70/30, and 50/50).

---

## 🛠️ 5. Tools & Technologies
- **Language:** Python 3.8+
- **Libraries:** Pandas, NumPy, Scikit-Learn.
- **Visualization:** Matplotlib, Seaborn.
- **Environment:** Jupyter Notebook / VS Code.

---

## 🚀 6. How to Run the Project
1. **Clone the repository:**
   ```bash
   git clone (https://github.com/mohamedkhaledmahmoud97-ux/Body-Performance-ML.git)
