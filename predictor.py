import pandas as pd
from ta import add_all_ta_features
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

# Load data
df = pd.read_csv(r"C:\Users\squii\Desktop\PythonApplication1\onehourbtc.csv")

# Ensure DataFrame is in chronological order
df.sort_values(by='date', ascending=True, inplace=True)

# Add all ta features filling mean values
df = add_all_ta_features(df, open="open", high="high", low="low", close="close", volume="Volume BTC")

# Define features set
X = df[df.columns.difference(['date', 'open', 'close', 'low', 'high'])]

# Define target variable
y = df['close']

# Split the data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create an imputer object that uses the 'mean' strategy to replace missing values
imputer = SimpleImputer(strategy='mean')

# Train the imputer on the features data
imputer.fit(X_train)

# Transform the training and test data using the trained imputer
X_train = pd.DataFrame(imputer.transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# Create a Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Get the model predictions
predictions = model.predict(X_test)

# Print out predictions
print(predictions)
