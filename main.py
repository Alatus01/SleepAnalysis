import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()

api.authenticate(username = 'ayushktripathi', key = '870eb3bbf37b58398bf7252e099651ec')

dataset_id = 'uom190346a/sleep-health-and-lifestyle-dataset'
api.dataset_download_files(dataset_id, unzip=True)

data = pd.read_csv('SleepHealthLifestyle.csv')

data.fillna(data.mean(), inplace=True)

# Scatter plot of Sleep Duration vs Age
plt.figure(figsize=(8, 6))
sns.scatterplot(data=data, x='Age', y='Sleep duration', alpha=0.5)
plt.title('Sleep Duration vs Age')
plt.xlabel('Age')
plt.ylabel('Sleep Duration')
plt.show()

# Linear regression model to predict sleep duration based on age
X = data[['Age']]
y = data['Sleep duration']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Plotting regression line
plt.figure(figsize=(8, 6))
sns.scatterplot(data=data, x='Age', y='Sleep duration', alpha=0.5)
plt.plot(X_train, model.predict(X_train), color='red')
plt.title('Linear Regression: Sleep Duration vs Age')
plt.xlabel('Age')
plt.ylabel('Sleep Duration')
plt.show()

# Calculate the optimized value of sleep duration
optimized_sleep_duration = model.predict([[30]])[0]
print("Optimized sleep duration for age 30:", optimized_sleep_duration)
