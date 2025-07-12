import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_data(file_path: str) -> pd.DataFrame:
    """ Load data from a CSV file. """
    data= pd.read_csv(file_path)
    print(data.shape)
    return data

def preprocess_and_split_data(data: pd.DataFrame, target_col: str, test_size: float = 0.2, random_state: int = 42) -> tuple:
    """
    Preprocess the data by:
    
    """
    X = data.drop(columns=[target_col])
    y = data[target_col]

    scaler = StandardScaler()
    numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    categorical_cols = X.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_features = encoder.fit_transform(X[categorical_cols])
        encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_cols), index=X.index)
        X = pd.concat([X.drop(columns=categorical_cols), encoded_df], axis=1)

    print(X.head())
    print(y.head())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test

def train_random_forest(X_train: pd.DataFrame, y_train: pd.DataFrame, n_estimators: int = 100, random_state: int = 42):  # Fixed: changed y_train type from DataFrame to Series
    """
    Train a Random Forest classifier.
    """
    from sklearn.ensemble import RandomForestClassifier
    if y_train.dtype == 'float':
        y_train = y_train.astype(int)
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series, n_estimators: int = 100, random_state: int = 42):  # Fixed: changed y_train type from DataFrame to Series
    """Train an XGBoost classifier. """
    X_train = X_train.select_dtypes(include=[np.number])
    X_train = X_train.fillna(0)
    y_train = pd.Series(y_train).astype(int)
    
    common_indices = X_train.index.intersection(y_train.index)
    X_train = X_train.loc[common_indices]
    y_train = y_train.loc[common_indices]
    
    model = XGBClassifier(objective='binary:logistic')
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.DataFrame) -> dict:  # Fixed: changed y_test type from DataFrame to Series
    """
    Evaluate the trained model on the test set.
    """
    
    if hasattr(model, 'get_booster'): 
        X_test = X_test.select_dtypes(include=[np.number]).fillna(0)
    
    y_pred = model.predict(X_test)
    y_pred = y_pred.astype(int)
    y_test = y_test.astype(int)
    print( accuracy_score(y_test, y_pred),
        precision_score(y_test, y_pred),
        recall_score(y_test, y_pred),
        f1_score(y_test, y_pred))
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }
    return metrics