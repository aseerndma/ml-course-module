import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# 1. Load the data
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
df = pd.read_csv(url)
df = df[["survived", "pclass", "sex", "age", "fare"]].copy()

# 2. Features and label
X = df[["pclass", "sex", "age", "fare"]]
y = df["survived"]

# 3. Split FIRST, before any preprocessing (prevents leakage)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Define how to preprocess each type of column
numeric_features = ["pclass", "age", "fare"]
categorical_features = ["sex"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),  # fill missing values
    ("scaler", StandardScaler())                    # standardize scale
])

categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))  # text -> numbers
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# 5. Bundle preprocessing + model into ONE pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# 6. Train the entire pipeline in one call
model.fit(X_train, y_train)

# 7. Evaluate
preds = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, preds), 3))
print("F1 score:", round(f1_score(y_test, preds), 3))

# 8. Save the whole fitted pipeline to disk
joblib.dump(model, "outputs/titanic_pipeline.joblib")
print("\nPipeline saved to outputs/titanic_pipeline.joblib")