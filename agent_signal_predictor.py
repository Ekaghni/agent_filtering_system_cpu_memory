import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('output.csv')

# Split the dataset into features (X) and target (y)
X = df[['cpu_usage', 'memory_usage']]
y = df['signal']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the Random Forest classifier
model = RandomForestClassifier(random_state=42)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy',n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Get the best model from hyperparameter tuning
best_model = grid_search.best_estimator_

# Save the best model to a file
joblib.dump(best_model, 'best_model.joblib')

# Save the scaler used during training
joblib.dump(scaler, 'scaler.joblib')

# Print accuracy on the training set
train_pred = best_model.predict(X_train_scaled)
train_accuracy = accuracy_score(y_train, train_pred)
print(f"Training Accuracy: {train_accuracy:.2%}")

# Print accuracy on the test set
test_pred = best_model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, test_pred)
print(f"Test Accuracy: {test_accuracy:.2%}")

# Optionally, plot feature importances
feature_importances = best_model.feature_importances_
feature_names = X.columns
plt.barh(feature_names, feature_importances)
plt.xlabel('Feature Importance')
plt.title('Feature Importances')
plt.show()
