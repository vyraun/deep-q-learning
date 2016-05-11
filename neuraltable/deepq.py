# import os
# os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"
# import theano

# import the neural net stuff
from keras.models import Sequential
from keras import optimizers
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.advanced_activations import LeakyReLU

# import other stuff
import random
import numpy as np

from memory import Memory

class DeepQ:
    def __init__(self, environment, inputs):
        self.input_size = inputs
        self.output_size = environment.action_space.n
        self.memory = Memory(10000)
        self.discountFactor = 0.975
        self.learnStart = 36
   
    def initNetwork(self, hiddenLayers):
        model = self.createModel(self.input_size, self.output_size, hiddenLayers, "relu", 0.01)
        self.model = model

    def createModel(self, inputs, outputs, hiddenLayers, activationType, learningRate):
        model = Sequential()
        if len(hiddenLayers) == 0: 
            model.add(Dense(self.output_size, input_shape=(self.input_size,), init='lecun_uniform'))
            model.add(Activation("linear"))
        else :
            model.add(Dense(hiddenLayers[0], input_shape=(self.input_size,), init='lecun_uniform'))
            
            if (activationType == "LeakyReLU") :
                model.add(LeakyReLU(alpha=0.01))
            else :
                model.add(Activation(activationType))
            
            for index in range(1, len(hiddenLayers)-1):
                layerSize = hiddenLayers[index]
                model.add(Dense(layerSize, init='lecun_uniform'))
                if (activationType == "LeakyReLU") :
                    model.add(LeakyReLU(alpha=0.01))
                else :
                    model.add(Activation(activationType))
            model.add(Dense(self.output_size, init='lecun_uniform'))
            model.add(Activation("linear"))
        optimizer = optimizers.RMSprop(lr=learningRate, rho=0.9, epsilon=1e-06)
        model.compile(loss="mse", optimizer=optimizer)
        return model

    def backupNetwork(self, model, backup):
        weightMatrix = []
        for layer in self.model.layers:
            weights = layer.get_weights()
            weightMatrix.append(weights)
        i = 0
        for layer in self.secondBrain.layers:
            weights = weightMatrix[i]
            layer.set_weights(weights)
            i += 1

    # predict Q values for all the actions
    def getQValues(self, state):
        predicted = self.model.predict(state.reshape(1,len(state)))
        return predicted[0]

    def getMaxQ(self, qValues=None):
        if (qValues is None):
            qValues = self.getQValues(state)
        return np.max(qValues)

    def getMaxIndex(self, qValues=None):
        if (qValues is None):
            qValues = self.getQValues(state)
        return np.argmax(qValues)

    # calculate the target function
    def calculateTarget(self, qValuesNewState, reward, isFinal):
        if isFinal:
            return reward
        else : 
            return reward + self.discountFactor * self.getMaxQ(qValuesNewState)

    # select the action with the highest Q value
    def selectAction(self, qValues, explorationRate):
        rand = random.random()
        if rand < explorationRate :
            action = np.random.randint(0, self.output_size)
        else :
            action = self.getMaxIndex(qValues)
        return action

    def selectActionByProbability(self, qValues, bias):
        qValueSum = 0
        shiftBy = 0
        for value in qValues:
            if value + shiftBy < 0:
                shiftBy = - (value + shiftBy)
        shiftBy += 1e-06

        for value in qValues:
            qValueSum += (value + shiftBy) ** bias

        probabilitySum = 0
        qValueProbabilities = []
        for value in qValues:
            probability = ((value + shiftBy) ** bias) / float(qValueSum)
            qValueProbabilities.append(probability + probabilitySum)
            probabilitySum += probability
        qValueProbabilities[len(qValueProbabilities) - 1] = 1

        rand = random.random()
        i = 0
        for value in qValueProbabilities:
            if (rand <= value):
                return i
            i += 1

    def addMemory(self, state, action, reward, newState, isFinal):
        self.memory.addMemory(state, action, reward, newState, isFinal)

    def learnOnLastState(self):
        if self.memory.getCurrentSize() >= 1:
            return self.memory.getMemory(self.memory.getCurrentSize() - 1)

    def learnOnMiniBatch(self, miniBatchSize): 
        if self.memory.getCurrentSize() > self.learnStart :
            miniBatch = self.memory.getMiniBatch(miniBatchSize)
            X_batch = np.empty((0,self.input_size), dtype = np.float64)
            Y_batch = np.empty((0,self.output_size), dtype = np.float64)
            for sample in miniBatch:
                isFinal = sample['isFinal']
                state = sample['state']
                action = sample['action']
                reward = sample['reward']
                newState = sample['newState']

                qValues = self.getQValues(state)
                qValuesNewState = self.getQValues(newState)
                targetValue = self.calculateTarget(qValuesNewState, reward, isFinal)

                X_batch = np.append(X_batch, np.array([state]), axis=0)
                Y_sample = qValues.copy()
                Y_sample[action] = targetValue
                Y_batch = np.append(Y_batch, np.array([Y_sample]), axis=0)
            self.model.fit(X_batch, Y_batch, batch_size = len(miniBatch), verbose = 0)


