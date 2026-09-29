import argparse

#getting the path of the current file
from pathlib import Path

# scores
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate

#sourcing the functions from the src folder
from src.data_loader import load_data, split_data
from src.models import choose_model

parser = argparse.ArgumentParser()

parser.add_argument('--threshold', type=float, default=0.3, help='Threshold for classification (default: 0.3)')
parser.add_argument('--model_name', type=str, default='svm', help='Picking a model (default: svm)')
parser.add_argument('--data_path', type=Path, default=Path('data/Heart_Disease_Cleveland.csv'), help='Path to dataset (default: data/Heart_Disease_Cleveland.csv)')

args = parser.parse_args()

# 0.picking the model to train
print("Starting training script...")

if args.model_name not in ["svm", "random_forest"]:
    raise ValueError("Invalid model name. Choose 'svm' or 'random_forest'.")

if args.model_name == "svm":
    print("You have chosen SVM model.")
elif args.model_name == "random_forest":
    print("You have chosen Random Forest model.")

n_estimators = 100  # Default value
max_depth = None  # Default value
min_samples_split = 2  # Default value

if args.model_name == "svm":
    print("Do you want to apply PCA for dimensionality reduction? (yes/no)")
    apply_pca = input().lower() == "yes"

if args.model_name == "random_forest":

    apply_pca = False  # PCA is not applied for Random Forest
    
    n_estimators_input = input("Enter the number of estimators for Random Forest (default is 100): ")

    if n_estimators_input:
        try:
            n_estimators = int(n_estimators_input)
        except ValueError:
            raise ValueError("Number of estimators must be an integer.")

    max_depth_input = input("Enter the maximum depth for Random Forest (default is None), and the range is [0, inf): ")

    if max_depth_input :
        try:
            max_depth = int(max_depth_input)
        except ValueError:
            raise ValueError("Maximum depth must be a positive integer.")
        else:
            if max_depth <= 0:
                raise ValueError("Maximum depth must be a positive integer.")

    min_samples_split_input = input("Enter the minimum samples split for Random Forest (default is 2): ")

    if min_samples_split_input:
        try:
            min_samples_split = int(min_samples_split_input)
        except ValueError:
            raise ValueError("Minimum samples split must be an integer.")
        else:
            if min_samples_split < 2:
                raise ValueError("Minimum samples split must be at least 2.")

# 1. Define file paths
DATA_PATH = Path("data") / "Heart_Disease_Cleveland.csv"

# asking for the categorical columns from the user
categorical_columns_input = input("Enter the categorical columns separated by commas (or leave blank if none): ")
categorical_columns = [col.strip() for col in categorical_columns_input.split(",") if col.strip()] if categorical_columns_input else None

print(f"The Categorical columns you have chosen are: {categorical_columns}")

# 2. Load and split dataset
X, y = load_data(DATA_PATH, categorical_columns=categorical_columns)
X_train, X_test, y_train, y_test = split_data(X, y)

# 3. Instantiate and fit model pipeline
print("Training model pipeline...")

pipeline = choose_model(args.model_name ,
                        apply_pca if args.model_name == "svm" else False,
                        n_estimators,
                        max_depth,
                        min_samples_split)

# 4. Evaluate with stratified cross-validation on the training set

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_validate(
    pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring=("roc_auc", "f1", "precision", "recall", "accuracy"),
)

print("5-fold cross-validation scores (mean +/- standard deviation):")

for metric in ("roc_auc", "f1", "precision", "recall", "accuracy"):
    scores = cv_scores[f"test_{metric}"]
    print(f"{metric}: {scores.mean():.4f} +/- {scores.std():.4f}")

# 5. Fit the final pipeline on all training data
pipeline.fit(X_train, y_train)

# 6. Predict probabilities and score on the held-out test set
print("Predicting probabilities...")

y_pred_proba = pipeline.predict_proba(X_test)[:,1]
y_pred_threshold = (y_pred_proba >= args.threshold).astype(int)

print(f"Model: {args.model_name} | PCA: {apply_pca if args.model_name == 'svm' else False} | Test ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
print(f"Model: {args.model_name} | PCA: {apply_pca if args.model_name == 'svm' else False} | Test f1 Score: {f1_score(y_test, y_pred_threshold):.4f}")
print(f"Model: {args.model_name} | PCA: {apply_pca if args.model_name == 'svm' else False} | Test Precision Score: {precision_score(y_test, y_pred_threshold):.4f}")
print(f"Model: {args.model_name} | PCA: {apply_pca if args.model_name == 'svm' else False} | Test Recall Score: {recall_score(y_test, y_pred_threshold):.4f}")
print(f"Model: {args.model_name} | PCA: {apply_pca if args.model_name == 'svm' else False} | Test Accuracy Score: {accuracy_score(y_test, y_pred_threshold):.4f}")
print(f"Model: {args.model_name} | PCA: {apply_pca if args.model_name == 'svm' else False} | Test Confusion Matrix: {confusion_matrix(y_test, y_pred_threshold)}")