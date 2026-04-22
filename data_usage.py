import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score , mean_absolute_error

# Step 1: Load dataset
df = pd.read_csv("data.csv")

# Rename columns
df = df.rename(columns={
    'Screen On Time (hours/day)': 'ScreenTime',
    'Data Usage (MB/day)': 'DataUsage',
    'App Usage Time (min/day)': 'AppUsage',
    'Battery Drain (mAh/day)': 'BatteryDrain',
    'Number of Apps Installed': 'AppsInstalled'
})

# Select relevant columns
df = df[['ScreenTime', 'AppUsage', 'BatteryDrain', 'AppsInstalled', 'DataUsage']]

# Remove Missing Values
df = df.dropna()

# Step 2: Define X and y
X = df[['ScreenTime', 'AppUsage', 'BatteryDrain', 'AppsInstalled']]
y = df['DataUsage']

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Predictions
y_pred = model.predict(X_test)

# Step 6: Accuracy
score = r2_score(y_test, y_pred)
print(f"\nR² Score: {round(score, 4)}")
print(f"Intercept: {model.intercept_:.2f}")
mae = mean_absolute_error(y_test, y_pred)
print(f"MAE: {mae:.2f}")

# Step 8: Test Case 
test_data = pd.DataFrame([{
    'ScreenTime': 5.5,
    'AppUsage': 220,
    'BatteryDrain': 1100,
    'AppsInstalled': 60
}])

prediction = model.predict(test_data)


print(f"\nPredicted Data Usage: {prediction[0]:.2f} MB/day")
print(f"Predicted Data Usage: {prediction[0]/1024:.2f} GB/day")

# Visualization
plt.scatter(y_test, y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted Data Usage")
plt.show()