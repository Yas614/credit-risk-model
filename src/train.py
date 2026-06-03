import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import mlflow
import mlflow.sklearn

# --- Helper Feature Lists & Pipeline Building Blocks ---
numerical_features = [
    "Amount", "Value", "hour", "day", "month", "year",
    "total_amount", "avg_amount", "std_amount", "transaction_count"
]
categorical_features = [
    "ProductCategory", "ChannelId", "ProviderId", "PricingStrategy"
]

def get_preprocessor():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("num", numeric_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ])


# --- Safe Execution Block ---
if __name__ == "__main__":
    # 1. Load Data
    df = pd.read_csv("data/processed/processed_data.csv")

    # 2. Separate Features and Target
    X = df.drop(columns=["is_high_risk"])
    y = df["is_high_risk"]

    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Initialize and Fit Preprocessing Pipeline
    preprocessor = get_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # 5. Train the Models
    print("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_processed, y_train)

    print("Training Random Forest...")
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train_processed, y_train)

    # 6. Evaluate
    y_pred_rf = rf.predict(X_test_processed)
    accuracy = accuracy_score(y_test, y_pred_rf)
    print(f"Model Accuracy: {accuracy:.4f}")

    # 7. Log to MLflow
    mlflow.set_experiment("Credit_Risk_Classification")

    with mlflow.start_run():
        # Log Parameters
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", rf.n_estimators)
        mlflow.log_param("random_state", 42)
        
        # Log Metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision_score(y_test, y_pred_rf))
        mlflow.log_metric("recall", recall_score(y_test, y_pred_rf))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred_rf))
        
        # Log the Model Object
        mlflow.sklearn.log_model(rf, "random_forest_model")
        
        print("Successfully logged run to MLflow!")