import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset directly from a public source
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
df = pd.read_csv(url)

# First look at the data
print("Shape (rows, columns):", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst 5 rows:\n", df.head())
print("\nMissing values per column:\n", df.isnull().sum())
# Summary 1: overall numeric description
print("\n--- Summary 1: numeric overview ---")
print(df.describe())

# Summary 2: survival rate by passenger class
print("\n--- Summary 2: survival rate by class ---")
print(df.groupby("class")["survived"].mean())

# Summary 3: average age by sex
print("\n--- Summary 3: average age by sex ---")
print(df.groupby("sex")["age"].mean())
# Plot 1: distribution of passenger ages
plt.figure()
df["age"].plot(kind="hist", bins=20, title="Age distribution of passengers")
plt.xlabel("Age")
plt.savefig("age_distribution.png")
plt.close()

# Plot 2: survival rate by class
plt.figure()
df.groupby("class")["survived"].mean().plot(kind="bar", title="Survival rate by class")
plt.ylabel("Survival rate")
plt.savefig("survival_by_class.png")
plt.close()

print("\nDone. Two plots saved as PNG files in this folder.")