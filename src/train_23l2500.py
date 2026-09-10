print("House Price Prediction Model - 2312500")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


# Load dataset
df = pd.read_csv("data/housing.csv")

# Remove unnecessary columns
df = df.drop(columns=["sr", "Order"], errors="ignore")

print(df.head())
print(df.info())

# Display target column
print("\nSale Prices:")
print(df["SalePrice"].head())

# Remove ID column
df = df.drop(columns="PID")

# Fill missing values
for column in df.columns:
    if df[column].dtype == "object":
        df[column] = df[column].fillna("None")
    else:
        df[column] = df[column].fillna(df[column].median)

# Convert categorical columns into numerical values
data = pd.get_dummies(df, dtype=int)

# Separate features and target
X = data.drop(columns="SalePrice")
y = data["SalePrice"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# Create the model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    max_features="sqrt"
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

# Save the trained model
joblib.dump(model, "model/house_price_model_23l2500.pkl")

print("\nModel saved successfully!")