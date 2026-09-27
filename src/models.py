from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from src.preprocessing import build_preprocessor


def choose_model(model_name: str,
                 apply_pca: bool,
                 n_estimators_input: int = 100,
                 max_depth_input: int = 0,
                 min_samples_split_input: int = 2) -> Pipeline:

    
    """Choose and return a machine learning pipeline based on the provided model name."""

    # 1. Select the classifier object based on model_name
    if model_name == "svm":

        classifier = SVC(probability=True, random_state=42)
        
    elif model_name == "random_forest":

        classifier = RandomForestClassifier(n_estimators=n_estimators_input,
                                            max_depth=max_depth_input,
                                            min_samples_split=min_samples_split_input)
    else:
        raise ValueError(
            "Invalid model name. Choose 'svm' or 'random_forest'."
        )

    # 2. Construct and return a single Pipeline
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("pca", PCA(n_components=5) if apply_pca else None),
            ("classifier", classifier),
        ]
    )
