import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load data
data = pd.read_csv("path_to_dataset.csv")

# Separate features and target
X = data.drop(columns=['Cover_Type'])
y = data['Cover_Type'] - 1  # Adjust labels if needed to start from 0

# Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Standardize numerical features (optional for tree-based models but often helpful)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Initialize Random Forest classifier
rf_model = RandomForestClassifier(random_state=42)

# Fit the model
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
print("Random Forest Classifier Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=[
    "Spruce/Fir", "Lodgepole Pine", "Ponderosa Pine", 
    "Cottonwood/Willow", "Aspen", "Douglas-fir", "Krummholz"
]))

# Confusion Matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Hyperparameter tuning with Grid Search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30, None],
    'max_features': ['sqrt', 'log2', None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid,
                           cv=3, scoring='accuracy', verbose=2, n_jobs=-1)

# Fit grid search
grid_search.fit(X_train, y_train)
best_rf_model = grid_search.best_estimator_

# Make predictions with the best model
y_best_pred = best_rf_model.predict(X_test)

# Final evaluation
print("Best Random Forest Model Accuracy:", accuracy_score(y_test, y_best_pred))
print("Best Model Classification Report:\n", classification_report(y_test, y_best_pred))

# Feature Importance
import matplotlib.pyplot as plt

feature_importances = best_rf_model.feature_importances_
indices = np.argsort(feature_importances)[::-1]

# Plot the feature importances of the forest
plt.figure(figsize=(12, 6))
plt.title("Feature Importance in Random Forest Classifier")
plt.bar(range(X.shape[1]), feature_importances[indices], align="center")
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.show()
