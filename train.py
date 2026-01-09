import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Paths
DATA_PATH = "dataset/winequality-red.csv"
MODEL_PATH = "output/model/model.pkl"
RESULTS_PATH = "output/results/metrics.json"

os.makedirs("output/model", exist_ok=True)
os.makedirs("output/results", exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH, sep=";")
X = df.drop("quality", axis=1)
y = df["quality"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print metrics
print(f"MSE: {mse}")
print(f"R2 Score: {r2}")

# Save outputs
joblib.dump(model, MODEL_PATH)
with open(RESULTS_PATH, "w") as f:
    json.dump({"mse": mse, "r2_score": r2}, f, indent=4)
