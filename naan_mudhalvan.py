# nan_muthalvan.py

# Cell 1: Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Cell 2: Load the Local Dataset
df = pd.read_csv('dataset_')  # Ensure this file is in the same folder
print("✅ Dataset Loaded Successfully")

# Cell 3: Display First 5 Rows
print("\n🔍 First 5 rows:\n", df.head())

# Cell 4: Dataset Info
print("\n📋 Dataset Info:")
print(df.info())

# Cell 5: Check for Missing Values
print("\n❓ Missing Values:\n", df.isnull().sum())

# Cell 6: Dataset Statistics
print("\n📊 Descriptive Stats:\n", df.describe())

# Cell 7: Correlation Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Cell 8: Target Variable Distribution (using last column as target)
target_column = df.columns[-1]
sns.histplot(df[target_column], bins=30, kde=True)
plt.title(f'Distribution of {target_column}')
plt.xlabel(target_column)
plt.ylabel('Frequency')
plt.show()

# Cell 9: Boxplot for Numeric Features
for column in df.select_dtypes(include=np.number).columns:
    plt.figure()
    sns.boxplot(x=df[column])
    plt.title(f'Boxplot for {column}')
    plt.show()

# Cell 10: Remove Outliers from One Feature (Example: First Numeric Column)
feature = df.select_dtypes(include=np.number).columns[0]
Q1 = df[feature].quantile(0.25)
Q3 = df[feature].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df[feature] < (Q1 - 1.5 * IQR)) | (df[feature] > (Q3 + 1.5 * IQR)))]
print(f"\n🧹 Outliers removed from {feature}")

# Cell 11: Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = df.drop(target_column, axis=1)
y = df[target_column]
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

# Cell 12: Train-Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("📦 Data split into training and test sets")

# Cell 13: Train Linear Regression Model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
print("✅ Linear Regression model trained")

# Cell 14: Evaluate Model
from sklearn.metrics import mean_squared_error, r2_score
y_pred = model.predict(X_test)
print("\n📈 Model Evaluation:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))

# Cell 15: Plot Actual vs Predicted
plt.scatter(y_test, y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted Values")
plt.show()
