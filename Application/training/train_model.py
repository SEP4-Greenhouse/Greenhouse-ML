# Application/training/train_dummy_model.py

from sklearn.linear_model import LogisticRegression
import joblib
import os

# Train dummy model
X = [[20, 25], [40, 50]]
y = [1, 0]

model = LogisticRegression()
model.fit(X, y)

# Save path (relative to this script)
output_path = os.path.join(os.path.dirname(__file__), "../ml_model/model.pkl")
output_path = os.path.abspath(output_path)

os.makedirs(os.path.dirname(output_path), exist_ok=True)
joblib.dump(model, output_path)

print(f"Model saved to: {output_path}")
