import pytest
from sklearn.ensemble import RandomForestClassifier
from ml.model import train_model, compute_model_metrics, inference
from ml.data import process_data
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Mock the dataset for testing
data = pd.read_csv('data/census.csv') 
train, test = train_test_split(data, test_size=0.2, random_state=42)
cat_features = [
    "workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"
]

# Run the data processing step (you can mock the result if needed)
X_train, y_train, encoder, lb = process_data(train, categorical_features=cat_features, label="salary", training=True)
X_test, y_test, _, _ = process_data(test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb)

# TODO: implement the first test. Check if the model is a RandomForestClassifier
def test_model_type():
    """
    Test if the model returned by train_model is of the correct type (RandomForestClassifier).
    """
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier), "Model should be an instance of RandomForestClassifier"

# TODO: implement the second test. Check if the inference function returns numpy array of correct shape
def test_inference_type_and_shape():
    """
    Test if the inference function returns predictions as a numpy array with the correct shape.
    """
    model = train_model(X_train, y_train)
    preds = inference(model, X_test)
    assert isinstance(preds, np.ndarray), "Predictions should be a numpy array"
    assert preds.shape[0] == X_test.shape[0], "Predictions should match the number of test samples"

# TODO: implement the third test. Check if metrics are between 0 and 1
def test_compute_model_metrics():
    """
    Test if compute_model_metrics returns precision, recall, and F1-score between 0 and 1.
    """
    model = train_model(X_train, y_train)
    preds = inference(model, X_test)
    precision, recall, f1 = compute_model_metrics(y_test, preds)
    assert 0 <= precision <= 1, f"Precision should be between 0 and 1, but got {precision}"
    assert 0 <= recall <= 1, f"Recall should be between 0 and 1, but got {recall}"
    assert 0 <= f1 <= 1, f"F1-score should be between 0 and 1, but got {f1}"

