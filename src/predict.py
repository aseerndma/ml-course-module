import joblib
import pandas as pd

# Load the saved pipeline
model = joblib.load("outputs/titanic_pipeline.joblib")

# A made-up passenger: 1st class, female, age 30, fare 100
new_passenger = pd.DataFrame([{
    "pclass": 1, "sex": "female", "age": 30, "fare": 100
}])

prediction = model.predict(new_passenger)[0]
print("Survived" if prediction == 1 else "Did not survive")