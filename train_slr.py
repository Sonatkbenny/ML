import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("student_data.csv")

# Handle missing values (replace with median)
df.fillna(df.median(numeric_only=True), inplace=True)

# Define independent (X) and dependent (Y) variables
X = df[['student_mark']]  # Independent variable
Y = df['student_grade']   # Dependent variable

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train the Simple Linear Regression model
model = LinearRegression()
model.fit(X_train, Y_train)

# Save the trained model
with open("model_slr.pkl", "wb") as file:
    pickle.dump(model, file)

print("✅ SLR model trained and saved as 'model_slr.pkl'!")
