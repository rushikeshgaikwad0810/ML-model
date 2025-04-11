import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import numpy as np

# Load dataset
df = pd.read_csv('Mall_Customers_100_with_Occupation.csv')

# Features & Target
X = df.drop(columns=['CustomerID', 'Spending Score (1-100)'])
y = df['Spending Score (1-100)']

# Preprocessing
numeric_features = ['Age', 'Annual Income (k$)']
categorical_features = ['Gender', 'Occupation']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(), categorical_features)
])

# Pipeline
pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
pipeline.fit(X_train, y_train)

# Predict & Evaluate
y_pred = pipeline.predict(X_test)

# ✅ Fix RMSE calculation for older scikit-learn versions
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("RMSE:", rmse)
print("R² Score:", r2_score(y_test, y_pred))

# ✅ Save the entire pipeline to a .pkl file
joblib.dump(pipeline, 'model.pkl')
print("✅ Model saved as model.pkl")
