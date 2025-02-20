class neuralNetwork:
    def __init__(self, inputN, hiddenN, outputN, lr):
        self.inum = inputN
        self.hnum = hiddenN
        self.onum = outputN
        self.lr = lr

        # wih is Weights from Input to Hidden
        self.wih = np.array(np.random.rand(self.hnum, self.inum) - 0.5)
        # who is Weights from Hidden to Output
        self.who = np.array(np.random.rand(self.onum, self.hnum) - 0.5)

        # AF is activation function
        self.AF = lambda x:scipy.special.expit(x)

    def train(self, inputList, targetList):
        # Calculate each layer
        inputs = np.array([inputList]).T
        targets = np.array([targetList]).T
        hiddenInputs = np.dot(self.wih, inputs)
        hiddenOutputs = self.AF(hiddenInputs)
        finalInputs = np.dot(self.who, hiddenOutputs)
        finalOutputs = self.AF(finalInputs)

        # Calculate the error of Outputs
        outputErrors = targets - finalOutputs
        # Back Propagate to Hidden Layer
        hiddenErrors = np.dot(self.who.T, outputErrors)

        # delta Wjk = 2 * alpha * (Tk - Ok) * Ok * (1 - Ok)* Oj'
        # Update who, Weights from Hidden to Output
        self.who += self.lr * np.dot((outputErrors * finalOutputs * (1 - finalOutputs)), hiddenOutputs.T)
        # Update wih, Weights from Inputs to Hidden
        self.wih += self.lr * np.dot((hiddenErrors * hiddenOutputs * (1 - hiddenOutputs)), inputs.T)

    def construct(self, inputList):
        inputs = np.array([inputList]).T
        hiddenInputs = np.dot(self.wih, inputs)
        hiddenOutputs = self.AF(hiddenInputs)
        finalInputs = np.dot(self.who, hiddenOutputs)
        finalOutputs = self.AF(finalInputs)

        return finalOutputs



class neuralNetwork1:
    def __init__(self, inputN, hiddenN, outputN, lr):
        self.inum = inputN
        self.hnum = hiddenN
        self.onum = outputN
        self.lr = lr

        # wih is Weights from Input to Hidden
        self.wih = np.array(np.random.rand(self.hnum, self.inum) - 0.5)
        # who is Weights from Hidden to Output
        self.who = np.array(np.random.rand(self.onum, self.hnum) - 0.5)
        # print("Shape of self.who: " + str(np.shape(self.who)))

    def train(self, inputList, targetList):
        # Calculate each layer
        inputs = np.array([inputList]).T
        # print("Shape of inputs: " + str(np.shape(inputs)))
        targets = np.array([targetList]).T
        # print("Shape of targets: " + str(np.shape(targets)))
        hiddenInputs = np.dot(self.wih, inputs)
        hiddenOutputs = hiddenInputs
        finalInputs = np.dot(self.who, hiddenOutputs)
        finalOutputs = finalInputs

        # Calculate the error of Outputs
        outputErrors = targets - finalOutputs
        # Back Propagate to Hidden Layer
        hiddenErrors = np.dot(self.who.T, outputErrors)

        # delta Wjk = 2 * alpha * (Tk - Ok) * Ok * (1 - Ok)* Oj'
        # Update who, Weights from Hidden to Output
        self.who += self.lr * np.dot(outputErrors, hiddenOutputs.T)
        # Update wih, Weights from Inputs to Hidden
        self.wih += self.lr * np.dot(hiddenErrors, inputs.T)

    def construct(self, inputList):
        inputs = np.array([inputList]).T
        hiddenInputs = np.dot(self.wih, inputs)
        hiddenOutputs = hiddenInputs
        finalInputs = np.dot(self.who, hiddenOutputs)
        finalOutputs = finalInputs

        return finalOutputs