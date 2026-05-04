import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
import os
import warnings

warnings.filterwarnings('ignore')

if not os.path.exists('images'):
    os.makedirs('images')

# ==========================================
# 1. DATA LOADING & PREPROCESSING
# ==========================================
def load_and_clean_data(filepath):
    print("[INFO] Loading and cleaning data...")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"[ERROR] File {filepath} not found. Please ensure it's in the same directory.")
        return None
    
    # Standardize column names
    df.columns = df.columns.str.replace(' ', '_').str.replace('%', 'pct')
    
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    
    # Handle impossible physiological values
    if 'diastolic' in df.columns:
        df['diastolic'] = df['diastolic'].replace(0, df['diastolic'].median())
    if 'systolic' in df.columns:
        df['systolic'] = df['systolic'].replace(0, df['systolic'].median())
    if 'broad_jump_cm' in df.columns:
        df['broad_jump_cm'] = df['broad_jump_cm'].replace(0, df['broad_jump_cm'].median())
    
    return df

# ==========================================
# 2. FEATURE ENGINEERING
# ==========================================
def engineer_features(df):
    print("[INFO] Engineering new features...")
    # BMI = weight / (height in meters)^2
    df['BMI'] = df['weight_kg'] / ((df['height_cm'] / 100) ** 2)
    
    # Relative Strength = gripForce / weight
    df['Relative_Strength'] = df['gripForce'] / df['weight_kg']
    
    # Core Ratio = sit-ups / broad jump (Safe division)
    df['Core_Ratio'] = np.where(df['broad_jump_cm'] > 0, 
                                df['sit-ups_counts'] / df['broad_jump_cm'], 0)
    
    # Encode categorical variables
    le_gender = LabelEncoder()
    df['gender'] = le_gender.fit_transform(df['gender'])
    
    le_class = LabelEncoder()
    df['class'] = le_class.fit_transform(df['class'])
    
    return df, le_class

# ==========================================
# 3. MODEL TRAINING & EVALUATION
# ==========================================
def train_and_evaluate(X_train, X_test, y_train, y_test, target_names):
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=15),
        "Decision_Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
        "SVM": SVC(kernel='rbf', C=10, random_state=42),
        "MLP_Neural_Network": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42),
        "Gradient_Boosting": HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n[MODEL] Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results[name] = acc
        
        print(f"✅ {name} Accuracy: {acc*100:.2f}%")
        
        # Save Confusion Matrix
        plt.figure(figsize=(6,5))
        cm = confusion_matrix(y_test, preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=target_names, yticklabels=target_names)
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(f'images/{name}_confusion_matrix.png')
        plt.close()

    return results

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("="*60)
    print("🏋️ BODY PERFORMANCE ANALYTICS SYSTEM - HAYPER DIGI")
    print("="*60)
    
    data = load_and_clean_data('bodyPerformance.csv')
    
    if data is not None:
        data, le_class = engineer_features(data)
        
        X = data.drop('class', axis=1)
        y = data['class']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        target_classes = ['A', 'B', 'C', 'D']
        final_scores = train_and_evaluate(X_train_scaled, X_test_scaled, y_train, y_test, target_classes)
        
        print("\n" + "="*40)
        print("🏆 FINAL RANKING (Accuracy)")
        print("="*40)
        for model, score in sorted(final_scores.items(), key=lambda x: x[1], reverse=True):
            print(f"{model.replace('_', ' '):<20} : {score*100:.2f}%")
        print("="*40)