import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

# Load dataset
data = pd.read_csv("Movie_Rating_Prediction/IMDb Movies India.csv", encoding='latin1')

# Select important columns
data = data[['Genre', 'Director', 'Actor 1', 'Rating']]

# Remove missing values
data = data.dropna()

# Convert text columns to numbers
encoder = LabelEncoder()

data['Genre'] = encoder.fit_transform(data['Genre'])
data['Director'] = encoder.fit_transform(data['Director'])
data['Actor 1'] = encoder.fit_transform(data['Actor 1'])

# Features and target
X = data[['Genre', 'Director', 'Actor 1']]
y = data['Rating']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Error
error = mean_absolute_error(y_test, predictions)

print("Mean Absolute Error:", error)