# !################################
# Import necessary libraries
# !################################

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import *

# !################################
# Import necessary libraries
# !################################
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# !################################
# Data Loading
# !################################
# student_df = pd.read_csv("StudentsPerformance.csv")
poker_df = pd.read_csv('poker-hand-training-true.data')

# student_df.head()
# student_df.info()
poker_df.head()
poker_df.info()

poker_df.columns
poker_df[['S1', 'R1', 'S2', 'R2', 'S3', 'R3', 'S4', 'R4', 'S5', 'R5', 'ORD']]


# !################################
# Test Train Split
# !################################
# 1. Prepare data
X = poker_df[['S1', 'R1', 'S2', 'R2', 'S3', 'R3', 'S4', 'R4', 'S5', 'R5']]  # 5 features
y = poker_df['ORD']

# 2. Split data
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.11, random_state=42)
# 8:1:1 = train:val:test

# 3. Scale features (important for KNN!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# !################################
# Train KNN Model
# !################################
# 5. Train final CLASSIFICATION model 
# For rare classes like Royal Flush, use K=1 to get exact matches
knn_final = KNeighborsClassifier(n_neighbors=1)  # Use K=1 for exact matching
knn_final.fit(X_train_scaled, y_train)


# !################################
# Prediction and Evaluation
# !################################
# 6. Make predictions and evaluate
predictions = knn_final.predict(X_test_scaled)
print(f"\nFinal Classification Model (K=1 for exact matching):")
from sklearn.metrics import accuracy_score, classification_report
print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")

# 7. For making predictions on new data - use EXACT training data patterns!
# These are exact Royal Flush patterns from the training data:
new_data = pd.DataFrame({
    'S1': [1, 2, 4],      
    'R1': [10, 11, 1],    # Exact patterns from training
    'S2': [1, 2, 4],      
    'R2': [11, 13, 13],   
    'S3': [1, 2, 4],        
    'R3': [13, 10, 12],   
    'S4': [1, 2, 4],      
    'R4': [12, 12, 11],   
    'S5': [1, 2, 4],      
    'R5': [1, 1, 10]      # Exact patterns from training
})

new_data_scaled = scaler.transform(new_data)
new_predictions = knn_final.predict(new_data_scaled)
print("\nPredictions for new data:", new_predictions)
print("Expected: Royal Flush (class 9)")

# Show prediction probabilities
probabilities = knn_final.predict_proba(new_data_scaled)
print("\nPrediction probabilities:")
for i, (pred, prob_row) in enumerate(zip(new_predictions, probabilities)):
    print(f"Sample {i+1}: Predicted class {pred}, confidence = {prob_row[pred]:.3f}")

 
 
 
# !################################
# Debug with K = 5 
# !################################
# Let's try a different approach - use KNeighborsClassifier instead of Regressor
from sklearn.neighbors import KNeighborsClassifier

# Try classification instead of regression
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train_scaled, y_train)

# Make predictions with classifier
new_predictions_class = knn_classifier.predict(new_data_scaled)
probabilities = knn_classifier.predict_proba(new_data_scaled)

print("Classification predictions:", new_predictions_class)
print("Prediction probabilities for each class:")
for i, prob_row in enumerate(probabilities):
    print(f"Sample {i+1}:")
    for class_idx, prob in enumerate(prob_row):
        if prob > 0.001:  # Only show non-zero probabilities
            print(f"  Class {class_idx}: {prob:.4f}")
    print()

 

# !################################
# 1, 3. 5
# !################################
# Solution 1: Use smaller K to focus on closest matches
print("=== Testing different K values for classification ===")
k_values_small = [1, 3, 5]

for k in k_values_small:
    knn_test = KNeighborsClassifier(n_neighbors=k)
    knn_test.fit(X_train_scaled, y_train)
    pred = knn_test.predict(new_data_scaled)
    print(f"K={k}: Predictions = {pred}")

print("\n=== Let's also test with actual Royal Flush from training data ===")
# Use one of the actual Royal Flush examples we found
actual_royal = poker_df[poker_df['ORD'] == 9].iloc[0:2][['S1', 'R1', 'S2', 'R2', 'S3', 'R3', 'S4', 'R4', 'S5', 'R5']]
print("Testing with actual Royal Flush hands from dataset:")
print(actual_royal)

actual_royal_scaled = scaler.transform(actual_royal)
pred_actual = KNeighborsClassifier(n_neighbors=1).fit(X_train_scaled, y_train).predict(actual_royal_scaled)
print(f"Prediction for actual Royal Flush examples: {pred_actual}")