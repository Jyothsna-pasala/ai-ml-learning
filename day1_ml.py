import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Sample data
# data = {
#     "size": [500, 750, 1000, 1250, 1500],
#     "price": [50, 75, 100, 125, 150]
# }
data = {
    "size": [500, 750, 1000, 1250, 1500, 1800, 2000],
    "price": [50, 75, 100, 125, 150, 180, 200]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)

# Split data
X = df[["size"]]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
# prediction = model.predict([[1200]])
prediction = model.predict([[1600], [2100]])

# print("Predicted price for size 1200:", prediction[0])
print("Predicted price for size 1600:", prediction[0])
print("Predicted price for size 2100:", prediction[1])