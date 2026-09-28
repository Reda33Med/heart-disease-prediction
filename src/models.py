from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from src.preprocessing import build_preprocessor


def choose_model(model_name: str,
                 apply_pca: bool,
                 n_estimators: int = 100,
                 max_depth: int | None = None,
                 min_samples_split: int = 2) -> Pipeline:

    
    """Choose and return a machine learning pipeline based on the provided model name."""

    # 1. Select the classifier object based on model_name
    if model_name == "svm":

        classifier = CalibratedClassifierCV(
            estimator=SVC(random_state=42),
            ensemble=False,
        )

    elif model_name == "random_forest":

        classifier = RandomForestClassifier(n_estimators=n_estimators,
                                            max_depth=max_depth,
                                            min_samples_split=min_samples_split,
                                            random_state=42)
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
