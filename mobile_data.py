import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Step 1: Load dataset
df = pd.read_csv("data.csv")

df = df[['Screen On Time (hours/day)', 'Data Usage (MB/day)']]

df = df.rename(columns={
    'Screen On Time (hours/day)': 'ScreenTime',
    'Data Usage (MB/day)': 'DataUsage'
})

df = df.dropna()

# Step 2: Define X and y
X = df[['ScreenTime']]
y = df['DataUsage']

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Predictions
y_pred = model.predict(X_test)


# Step 6: Accuracy — R² Score
score = r2_score(y_test, y_pred)
print(f"  R² Score  : {round(score, 4)}  (closer to 1 = better)")

# Step 7: Model Equation

print("\nModel Equation:")
print(f"DataUsage = {model.coef_[0]:.2f} * ScreenTime + {model.intercept_:.2f}")

# Step 8: User Prediction 
while True:
    try:
        hours = float(input("\nEnter screen time (hours/day): "))
        if hours < 0:
            print("Screen time can't be negative. Please try again.")
        elif hours > 24:
            print("Screen time can't exceed 24 hours. Please try again.")
        else:
            break
    except ValueError:
        print("Invalid input. Please enter a number like 5 or 6.5")

input_data = pd.DataFrame([[hours]], columns=['ScreenTime'])
predicted_data = model.predict(input_data)

print(f"\nEstimated Data Usage: {predicted_data[0]:.2f} MB/day")
print(f"Estimated Data Usage: {predicted_data[0]/1024:.2f} GB/day")

# Step 9: Visualization  
plt.figure(figsize=(8, 5))

# Actual test values
plt.scatter(X_test, y_test, color='blue', label="Actual (Test Data)", alpha=0.6)

# What model predicted for those same test points
plt.scatter(X_test, y_pred, color='red', label="Predicted", marker='x', s=80)

# Regression line
plt.plot(X_test, y_pred, color='red', linewidth=1, alpha=0.4)

plt.xlabel("Screen Time (hours/day)")
plt.ylabel("Data Usage (MB/day)")
plt.title("Mobile Data Usage Prediction — Test Set Performance")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
