import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

diabetes = datasets.load_diabetes()

# print(diabetes.keys())
# (['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module'])

# print(diabetes.DESCR)

# diabetes_X = diabetes.data[:,np.newaxis,2]
# diabetes_X = diabetes.data
diabetes_X = np.array([[1],[2],[3]])
# print(diabetes_X)

# sliching
diabetes_X_train = diabetes_X
diabetes_X_test = diabetes_X

diabetes_Y_train = np.array([3,2,4])
diabetes_Y_test = np.array([3,2,4])


model = linear_model.LinearRegression()
model.fit(diabetes_X_train, diabetes_Y_train)

diabetes_Y_predicted = model.predict(diabetes_X_test)

print("Mean squared error is :", mean_squared_error(diabetes_Y_test,diabetes_Y_predicted))
print("Weights are :", model.coef_)
print("Intercepts are :", model.intercept_)

# Mean squared error is : 3035.060115291269
# Weights are : [941.43097333]
# Intercepts are : 153.39713623331644

plt.scatter(diabetes_X_test, diabetes_Y_test)
plt.plot(diabetes_X_test,diabetes_Y_predicted)
plt.show()