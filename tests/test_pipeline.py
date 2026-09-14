import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def load_data():
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
    df = pd.read_csv(url)
    return df[["survived", "pclass", "sex", "age", "fare"]].copy()


def build_model():
    preprocessor = ColumnTransformer(transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), ["pclass", "age", "fare"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["sex"])
    ])
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ])


# --- DATA TESTS ---

def test_data_has_expected_columns():
    df = load_data()
    expected = {"survived", "pclass", "sex", "age", "fare"}
    assert expected.issubset(df.columns), "A required column is missing"


def test_survived_is_binary():
    df = load_data()
    assert set(df["survived"].unique()).issubset({0, 1}), \
        "survived should only contain 0 or 1"


def test_age_is_reasonable():
    df = load_data()
    ages = df["age"].dropna()
    assert ages.min() >= 0, "age cannot be negative"
    assert ages.max() < 120, "age is unrealistically high"


# --- MODEL TEST ---

def test_model_beats_baseline():
    df = load_data()
    X = df[["pclass", "sex", "age", "fare"]]
    y = df["survived"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = build_model()
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    # Baseline: always guessing the majority class is ~0.62 here.
    assert acc > 0.7, f"Model accuracy {acc:.3f} is below acceptable threshold"