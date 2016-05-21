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
        self.memory = Memory(2000)
        self.discountFactor = 0.975
        self.learnStart = 36
        self.models = [None] * 5
   
    def initNetwork(self, hiddenLayers):
        model = self.createModel(self.input_size, self.output_size, hiddenLayers, "relu", 0.01)
        self.models[0] = model

        model2 = self.createModel(self.input_size, self.output_size, hiddenLayers, "relu", 0.01)
        self.models[1] = model2

        model3 = self.createModel(self.input_size, self.output_size, hiddenLayers, "relu", 0.01)
        self.models[2] = model3


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
    def getQValues(self, state, modelNr=0):
        predicted = self.models[modelNr].predict(state.reshape(1,len(state)))
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

    def selectActionMostConfident(self, qValues, qValues2, explorationRate):
        rand = random.random()
        if rand < explorationRate :
            action = np.random.randint(0, self.output_size)
        else :
            maxQ1 = self.getMaxQ(qValues)
            maxQ2 = self.getMaxQ(qValues2)
            if (abs(maxQ1) > abs(maxQ2)):
                action = self.getMaxIndex(qValues)
            else :
                action = self.getMaxIndex(qValues2)
        return action

    def selectActionAverage(self, qValues, qValues2, explorationRate):
        rand = random.random()
        if rand < explorationRate :
            action = np.random.randint(0, self.output_size)
        else :
            avgQValues = []
            for i in range(0, len(qValues)-1):
                value1 = qValues[i]
                value2 = qValues2[i]
                avg = (value1 + value2) / 2.0
                avgQValues.append(avg)
            action = self.getMaxIndex(avgQValues)
        return action

    def selectActionAdded(self, qValues, qValues2, explorationRate):
        rand = random.random()
        if rand < explorationRate :
            action = np.random.randint(0, self.output_size)
        else :
            addedQValues = qValues + qValues2
            action = self.getMaxIndex(addedQValues)
        return action

    def selectActionMostPreferred(self, qValues, qValues2, qValues3, explorationRate):
        rand = random.random()
        if rand < explorationRate :
            action = np.random.randint(0, self.output_size)
        else :
            action1 = self.getMaxIndex(qValues)
            action2 = self.getMaxIndex(qValues2)
            action3 = self.getMaxIndex(qValues3)
            actionsChosen = [0, 0]
            actionsChosen[action1] += 1
            actionsChosen[action2] += 1
            actionsChosen[action3] += 1
            if (actionsChosen[0] > actionsChosen[1]):
                action = 0
            else :
                action = 1
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

    def learnOnMiniBatch(self, miniBatchSize, modelNr=0): 
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
            self.models[modelNr].fit(X_batch, Y_batch, batch_size = 1, verbose = 0)


