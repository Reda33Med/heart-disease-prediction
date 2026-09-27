from pathlib import Path
from sklearn.metrics import roc_auc_score
from src.data_loader import load_data, split_data
from src.models import choose_model

# 0.picking the model to train
print("Starting training script...")
print("Choose model: 'svm' or 'random_forest'")
model_name = input("Enter model name: ")

if model_name not in ["svm", "random_forest"]:
    raise ValueError("Invalid model name. Choose 'svm' or 'random_forest'.")

if model_name == "svm":
    print("You have chosen SVM model.")
elif model_name == "random_forest":
    print("You have chosen Random Forest model.")


if model_name == "svm":
    print("Do you want to apply PCA for dimensionality reduction? (yes/no)")
    apply_pca = input().lower() == "yes"

if model_name == "random_forest":

    apply_pca = False  # PCA is not applied for Random Forest

    n_estimators_input = input("Enter the number of estimators for Random Forest (default is 100): ")

    if n_estimators_input:
        try:
            n_estimators = int(n_estimators_input)
        except ValueError:
            raise ValueError("Number of estimators must be an integer.")

    max_depth_input = input("Enter the maximum depth for Random Forest (default is None): ")

    if max_depth_input:
        try:
            max_depth = int(max_depth_input)
        except ValueError:
            raise ValueError("Maximum depth must be an integer.")

    min_samples_split_input = input("Enter the minimum samples split for Random Forest (default is 2): ")

    if min_samples_split_input:
        try:
            min_samples_split = int(min_samples_split_input)
        except ValueError:
            raise ValueError("Minimum samples split must be an integer.")

# 1. Define file paths
DATA_PATH = Path("data") / "Heart_Disease_Cleveland.csv"

# 2. Load and split dataset
X, y = load_data(DATA_PATH)
X_train, X_test, y_train, y_test = split_data(X, y)

# 3. Instantiate and fit model pipeline
print("Training model pipeline...")
pipeline = choose_model(model_name , apply_pca if model_name == "svm" else False)
pipeline.fit(X_train, y_train)

# 4. Predict probabilities and score
print("Predicting probabilities...")
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
score = roc_auc_score(y_test, y_pred_proba)

print(f"Model: {model_name} | PCA: {apply_pca if model_name == 'svm' else False} | Test ROC-AUC Score: {score:.4f}")