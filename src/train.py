import argparse
import yaml
import joblib
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Read command-line arguments
parser = argparse.ArgumentParser(description="Train the Titanic model.")
parser.add_argument("--config", default="configs/config.yaml",
                    help="Path to the config file")
args = parser.parse_args()

# 2. Load settings from the config file
with open(args.config, "r") as f:
    config = yaml.safe_load(f)

print(f"Loaded config: {config}")

# 3. Load and prepare data
df = pd.read_csv(config["data_url"])
df = df[["survived", "pclass", "sex", "age", "fare"]].copy()
X = df[["pclass", "sex", "age", "fare"]]
y = df["survived"]

# 4. Split using values FROM the config
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=config["test_size"],
    random_state=config["random_state"]
)

# 5. Build the pipeline
numeric_features = ["pclass", "age", "fare"]
categorical_features = ["sex"]

preprocessor = ColumnTransformer(transformers=[
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# 6. Train
model.fit(X_train, y_train)

# 7. Evaluate
preds = model.predict(X_test)
metrics = {
    "accuracy": round(accuracy_score(y_test, preds), 3),
    "precision": round(precision_score(y_test, preds), 3),
    "recall": round(recall_score(y_test, preds), 3),
    "f1": round(f1_score(y_test, preds), 3),
}

# 8. Save the model
joblib.dump(model, "outputs/titanic_pipeline.joblib")

# 9. Write a report file automatically
with open("outputs/report.md", "w") as f:
    f.write("# Training Report\n\n")
    f.write(f"Run at: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
    f.write(f"Config used: `{args.config}`\n\n")
    f.write("## Metrics\n\n")
    for name, value in metrics.items():
        f.write(f"- **{name}**: {value}\n")

print("Metrics:", metrics)
print("Model saved to outputs/titanic_pipeline.joblib")
print("Report saved to outputs/report.md")