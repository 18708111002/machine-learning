class LWLRegression:

    def __init__(self):
        self.X_data = None
        self.labels = None
        self.cntData= None

        self.k = 0.01


    def fittedData(self,x_data,labels):

        self.X_data = x_data
        self.labels = labels
        self.cntData = len(self.X_data)

    def hypothesis(self,x,theta):
        h = 0

        for(thetaList,x) in zip(theta,x):
            for theta in thetaList:
                h = h + theta * x

        return h

    def getWeight(self,x_i,x):
        index = (np.square(np.linalg.norm(x_i - x)) * -1.0) / (2 * self.k ** 2)
        return np.exp(index)

    def trainModel(self,x):

        W = np.zeros((self.cntData,self.cntData))
        for i in range(self.cntData):
            W[i][i] = self.getWeight(self.X_data[i],x)

        X_T = self.X_data.T

        part1 = np.mat(X_T) *  np.mat(W) * np.mat(self.X_data)
        # print(part1)
        theta = (part1.I) * np.mat(X_T) * np.mat(W) * np.mat(self.labels)
        # print(theta.reshape(len(theta),order='C'))


        # theta = [x for x in theta.reshape(len(theta),order='C')]

        return theta.tolist()

    def predict(self,x_test):
        pre_labes = []

        for x in x_test:
            theta = self.trainModel(x)
            pre_labes.append(self.hypothesis(x,theta))

        return pre_labes

if __name__ == '__main__':
    import numpy as np
    from sklearn import datasets
    from sklearn import linear_model

    # age sex bodyExp blood s1 s2 s3 s4 s5 s6
    diabetes=datasets.load_diabetes()

    x_train=[]
    y_train=[]

    with open('data.txt') as f:
        for line in f:
            x_train.append([float(line.split()[1])])
            y_train.append([float(line.split()[2])])

    x_train = np.array(x_train[0:20])
    y_train = np.array(y_train[0:20])

    x_test = x_train[0:20]
    y_test = y_train[0:20]

    linreg = LWLRegression()
    linreg.fittedData(x_train,y_train)


import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

plt.figure()

for f in [0]:

    plt.ylabel(u'table')
    print(sorted(x_test))
    x_test = sorted(x_test)
    y = linreg.predict(x_test)


    plt.scatter(x_test, y_test, color='k')

    plt.plot(x_test, y)

plt.show()

