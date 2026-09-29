from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor() -> ColumnTransformer:
    """Build and return the Scikit-Learn ColumnTransformer for numerical and categorical features."""
    # Pipeline for categorical columns
    categorical_pipeline = Pipeline(
        [
            ("imputer",SimpleImputer(strategy="constant", fill_value="missing")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    # Pipeline for numerical columns
    numerical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Combine into a single ColumnTransformer
    preprocessor = ColumnTransformer(
        [
            (
                "numerical",
                numerical_pipeline,
                make_column_selector(dtype_include= "number"),
            ),
            (
                "categorical",
                categorical_pipeline,
                make_column_selector(dtype_include=["object", "category"]),
            ),
        ]
    )

    return preprocessor