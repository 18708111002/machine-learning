#coding: utf-8
import numpy as np
from LeastMeanSquare import LMSRegression
from sklearn import datasets
from sklearn import linear_model

# age sex bodyExp blood s1 s2 s3 s4 s5 s6
diabetes=datasets.load_diabetes()

x_train=diabetes.data[:-20]
y_train=diabetes.target[:-20]
x_test=diabetes.data[-20:]
y_test=diabetes.target[-20:]




linreg = linear_model.LinearRegression();
linreg.fit(x_train,y_train)
print(linreg.coef_ )


import matplotlib.pyplot as plt
import matplotlib.font_manager as fm



plt.figure()

for f in range(0, 10):

    xi_test = x_test[:, f]

    xi_train = x_train[:, f]

    xi_test = xi_test[:, np.newaxis]
    xi_train = xi_train[:, np.newaxis]

    plt.ylabel(u'病情数值')

    y = linreg.predict(xi_test)

    plt.subplot(5, 2, f + 1)

    plt.scatter(xi_test, y_test, color='k')

    plt.plot(xi_test, y, color='b', linewidth=3)

plt.show()

