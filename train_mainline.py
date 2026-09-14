import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Load the data
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
df = pd.read_csv(url)

# 2. Prepare the data (simple version)
df = df[["survived", "pclass", "sex", "age", "fare"]].copy()
df["age"] = df["age"].fillna(df["age"].median())   # fill missing ages
df["sex"] = df["sex"].map({"male": 0, "female": 1})  # text -> numbers
df = df.dropna()  # drop any leftover missing rows

# 3. Split into features (X) and label (y)
X = df[["pclass", "sex", "age", "fare"]]
y = df["survived"]

# 4. Hold back 20% of the data for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Train two different models
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)

tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

# 6. Evaluate each model on the unseen test data
def evaluate(model, name):
    preds = model.predict(X_test)
    print(f"\n--- {name} ---")
    print("Accuracy: ", round(accuracy_score(y_test, preds), 3))
    print("Precision:", round(precision_score(y_test, preds), 3))
    print("Recall:   ", round(recall_score(y_test, preds), 3))
    print("F1 score: ", round(f1_score(y_test, preds), 3))

evaluate(logreg, "Logistic Regression")
evaluate(tree, "Decision Tree")