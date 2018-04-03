class LMSRegression:

    def __init__(self):
        self.alph = 30
        self.theta = []

        self.X_data = None;
        self.labes = None;

        self.count = 0

    def isConvergence(self):

        if(self.count < 100):
            self.count += 1

            return False

        return True

    def hypothesis(self,x):
        h = 0
        for(theta,x) in zip(self.theta,x):
            h = h + theta * x

        return h

    def initTheta(self,featuresNum):
        for i in range(featuresNum):
            self.theta.append(0)
        pass

    def train(self,x_data,labes):
        self.X_data = x_data
        self.labes = labes
        self.initTheta(len(x_data[0]))

        while(not self.isConvergence()):
            for i in range(len(self.X_data)):
                for j in range(len(self.theta)):
                    delta = (self.labes[i] - self.hypothesis(self.X_data[i])) * self.X_data[i][j]
                    self.theta[j] = self.theta[j] + self.alph * delta

    def predict(self,x_test):
        labels = []
        for x in x_test:
            labels.append(self.hypothesis(x))

        return labels

    def getTheta(self):
        return self.theta

if __name__ == '__main__':
    import numpy as np
    from LeastMeanSquare import LMSRegression
    from sklearn import datasets
    from sklearn import linear_model

    # age sex bodyExp blood s1 s2 s3 s4 s5 s6
    diabetes = datasets.load_diabetes()

    x_train = diabetes.data[:-20]
    y_train = diabetes.target[:-20]
    x_test = diabetes.data[-20:]
    y_test = diabetes.target[-20:]

    xt = x_train.T
    linreg = LMSRegression()
    linreg.train(x_train, y_train)

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
