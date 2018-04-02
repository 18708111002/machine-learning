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