import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (20.0, 10.0)

# Reading data
data = pd.read_csv("D:\headbrain.csv")
print(data.shape)
data.head()

# Collecting X and Y
X = data['Head Size(cm^3)'].values
Y = data['Brain Weight(grams)'].values

# Mean X and Y
mean_x = np.mean(X)
mean_y = np.mean(Y)

# Total data points 
n = len(X)

# Using formula to calculate b1 and b2
ntr = 0
dtr = 0

for i in range(n):
  ntr += (X[i] - mean_x) * (Y[i] - mean_y)
  dtr += (X[i] - mean_x) ** 2
b1 = ntr / dtr # slope(m)
b0 = mean_y - (b1 * mean_x) # y-intercept c

# Print Coefficients 
print("m: " , b1, "\nc: ", b0)

# m = 0.263, c = 325.573
# Brain_Weight = 0.263 * Head_Size + 325.573

# Plotting values and Regression Line
max_x = np.max(X) + 100
min_x = np.min(X) - 100

# Calculating line values x and y
x = np.linspace(min_x, max_x, 1000)
y = b0 + b1 * x

# Plotting Line
plt.plot(x, y, color = '#58b970', label = 'Regression')

# Plotting Scatter Points
plt.scatter(X, Y, c = '#ef5423', label = 'Scatter Plot')

plt.xlabel('Head Size in cm3')
plt.ylabel('Brain Weight in grams')
plt.legend()
plt.show()

ss_ntr = 0
ss_dtr = 0
for i in range(n):
  y_pred = b0 + b1 * X[i]
  ss_ntr += (y_pred - mean_y) ** 2
  ss_dtr += (Y[i] - mean_y) ** 2

r2 = ss_ntr / ss_dtr
print("r_square: ", r2)

# Now import the necessary libraries
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Sample data
X = np.array([1, 2, 3, 4, 5])
Y = np.array([1, 4, 9, 16, 25])

# Reshape X to be a 2D array
X = X.reshape(-1, 1)

# Creating the model
reg = LinearRegression()

# Fitting the model with training data
reg.fit(X, Y)

# Predicting Y using the trained model
Y_pred = reg.predict(X)

# Calculating R² Score
r2_score = reg.score(X, Y)

print(f'R² score: {r2_score}')

Confidence_level = r2_score * 100
print('Confidence level: ', Confidence_level, "%")