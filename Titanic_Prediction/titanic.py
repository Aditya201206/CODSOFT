import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("Titanic_Prediction/Titanic-Dataset.csv")
# Select useful columns
data = data[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']]

# Convert male/female to numbers
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

# Fill missing age values
data['Age'] = data['Age'].fillna(data['Age'].mean())

# Inputs and output
X = data[['Pclass', 'Sex', 'Age', 'Fare']]
y = data['Survived']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression()

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("Titanic Model Accuracy:", accuracy * 100)