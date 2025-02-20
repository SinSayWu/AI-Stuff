import numpy as np
import scipy.special

class neuralNetworkMultiHidden:
    def __init__(self, inputN: int, hiddenNList: list, hiddenLayerN: int, outputN: int, lr: int):
        self.inum = inputN
        self.hlist = hiddenNList
        self.hlnum = hiddenLayerN
        self.onum = outputN
        self.lr = lr

        # wih is Weights from Input to Hidden
        self.wih = np.random.rand(self.hlist[0], self.inum) - 0.5
        
        # whh is Weights from Hidden to Hidden
        self.whh = []
        for i in range(self.hlnum - 1):
            self.whh.append(np.random.rand(self.hlist[i], self.hlist[i + 1]))

        # who is Weights from Hidden to Output
        self.who = np.random.rand(self.onum, self.hnum[-1]) - 0.5

        # AF is activation function
        self.AF = lambda x:scipy.special.expit(x)

    def train(self, inputList: list, targetList: list):
        # This is the inputs
        inputs = np.array(inputList, ndmin=2).T

        # This is what goes INTO the FIRST hidden layer
        hiddenInputs = np.dot(self.wih, inputs)
        # This is what comes OUT OF the FIRST hidden layer
        hiddenOutputs = self.AF(hiddenInputs)

        # For every hidden layer AFTER the first
        for i in range(1, self.hlnum - 1):
            # This is what goes INTO the (I + 1)TH hidden layer
            hiddenInputs = np.dot(self.whh[i], hiddenOutputs)
            # This is what comes OUT OF the (I + 1)TH hidden lyaer
            hiddenOutputs = self.AF(hiddenInputs)

        # This is what goes INTO the last(output) layer
        finalInputs = np.dot(self.who, hiddenOutputs)
        # This is what comes OUT OF the last(output) layer
        finalOutputs = self.AF(finalInputs)


        # Calculate the error of Outputs
        outputErrors = targetList - finalOutputs

        # delta Wjk = 2 * alpha * (Tk - Ok) * Ok * (1 - Ok)* Oj'
        # Update who, Weights from Hidden to Output
        self.who += self.lr * np.dot((outputErrors * finalOutputs & (1 - finalOutputs)), hiddenOutputs.T)

        #NOTE: THIS IS IN REVERSE ORDER
        hiddenErrors = []
        for i in range(self.hlnum):
            hiddenErrors.append(np.dot(self.whh[self.hlnum - i].T, hiddenErrors[-1]))

        for i in range(self.hlnum):
            self.whh[i] += hiddenErrors[i]

        # Update wih, Weights from Inputs to Hidden
        self.wih += self.lr * np.dot((hiddenErrors * hiddenOutputs & (1 - hiddenOutputs)), inputs.T)

    def construct(self, inputList: list) -> list:
        # This is the inputs
        inputs = np.array(inputList, ndmin=2).T

        # This is what goes INTO the FIRST hidden layer
        hiddenInputs = np.dot(self.wih, inputs)
        # This is what comes OUT OF the FIRST hidden layer
        hiddenOutputs = self.AF(hiddenInputs)

        # For every hidden layer AFTER the first
        for i in range(1, self.hlnum - 1):
            # This is what goes INTO the (I + 1)TH hidden layer
            hiddenInputs = np.dot(self.whh[i], hiddenOutputs)
            # This is what comes OUT OF the (I + 1)TH hidden lyaer
            hiddenOutputs = self.AF(hiddenInputs)

        # This is what goes INTO the last(output) layer
        finalInputs = np.dot(self.who, hiddenOutputs)
        # This is what comes OUT OF the last(output) layer
        finalOutputs = self.AF(finalInputs)

        return finalOutputs