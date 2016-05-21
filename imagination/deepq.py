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

class Tree(object):
    def __init__(self):
        self.left = None
        self.right = None
        self.data = None


class DeepQ:
    def __init__(self, environment, inputs):
        self.environment = environment
        self.state_size = inputs
        self.nr_actions = environment.action_space.n
        self.memory = Memory(30000)
        self.discountFactor = 0.975
        self.predictionModels = []
   
    def initImaginationNetworks(self):
        for t in xrange(self.nr_actions):
            self.predictionModels.insert(t, self.createModel(self.state_size, self.state_size, [self.state_size, self.state_size, self.state_size], "relu", 0.01))

    def initRewardNetwork(self):
        self.rewardModel = self.createModel(self.state_size, 1, [self.state_size, self.state_size, self.state_size], "relu", 0.01)

    def createModel(self, inputs, outputs, hiddenLayers, activationType, learningRate):
        model = Sequential()
        if len(hiddenLayers) == 0: 
            model.add(Dense(outputs, input_shape=(inputs,), init='lecun_uniform'))
            model.add(Activation("linear"))
        else :
            model.add(Dense(hiddenLayers[0], input_shape=(inputs,), init='lecun_uniform'))
            
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
            model.add(Dense(outputs, init='lecun_uniform'))
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

    def getStatePrediction(self, state, action):
        predicted = self.predictionModels[action].predict(state.reshape(1,len(state)))
        return predicted[0]

    def getPredictedStates(self, state):
        predictedStates = []
        for a in xrange(self.nr_actions):
            predictedStates.insert(a, self.getStatePrediction(state, a))
        return predictedStates

    def getStateValuePrediction(self, state):
        predictedReward = self.rewardModel.predict(state.reshape(1,len(state)))
        return predictedReward[0][0]

    def getPredictedActionValues(self, state):
        predictedActionValues = []
        for a in xrange(self.nr_actions):
            predictedActionValues.insert(a, self.getStateValuePrediction(self.getStatePrediction(state, a)))
        return predictedActionValues

    def getMaxValue(self, array):
        return np.max(array)

    def getMaxIndex(self, array):
        return np.argmax(array)

    def getTarget(self, state, reward, isFinal):
        if isFinal:
            return reward
        else:
            predictedActionValues = self.getPredictedActionValues(state)
            # return reward + self.discountFactor * (sum(predictedActionValues)/len(predictedActionValues))
            return reward + self.discountFactor * np.max(predictedActionValues)


    def printStatePredictionTree(self, state):
        root = Tree()
        # first layer
        predicted1 = self.getPredictedStates(state)
        root.data = state
        root.left = Tree()
        root.left.data = predicted1[0]
        root.right = Tree()
        root.right.data = predicted1[1]

        # second layer
        predicted2left = self.getPredictedStates(predicted1[0])
        root.left.left = Tree()
        root.left.left.data = predicted2left[0]
        root.left.right = Tree()
        root.left.right.data = predicted2left[1]
        predicted2right = self.getPredictedStates(predicted1[1])
        root.right.left = Tree()
        root.right.left.data = predicted2right[0]
        root.right.right = Tree()
        root.right.right.data = predicted2right[1]

        print ""
        print "\t\t\t\t\t\t\t\t\t\t",root.data
        print "\t\t\t\t",root.left.data,"\t\t\t\t\t\t\t",root.right.data
        print root.left.left.data,"\t",root.left.right.data,"\t",root.right.left.data,"\t",root.right.right.data

    def printStateValueTree(self, state):
        root = Tree()
        # first layer
        predicted1 = self.getPredictedStates(state)
        root.data = state
        root.left = Tree()
        root.left.data = predicted1[0]
        root.right = Tree()
        root.right.data = predicted1[1]

        # second layer
        predicted2left = self.getPredictedStates(predicted1[0])
        root.left.left = Tree()
        root.left.left.data = predicted2left[0]
        root.left.right = Tree()
        root.left.right.data = predicted2left[1]
        predicted2right = self.getPredictedStates(predicted1[1])
        root.right.left = Tree()
        root.right.left.data = predicted2right[0]
        root.right.right = Tree()
        root.right.right.data = predicted2right[1]

        print ""
        print "\t\t\t\t\t\t\t\t\t\t",self.getStateValuePrediction(root.data)
        print "\t\t\t\t",self.getStateValuePrediction(root.left.data),"\t\t\t\t\t\t\t\t\t\t\t",self.getStateValuePrediction(root.right.data)
        print self.getStateValuePrediction(root.left.left.data),"\t\t\t\t\t",self.getStateValuePrediction(root.left.right.data),"\t\t\t\t\t",self.getStateValuePrediction(root.right.left.data),"\t\t\t\t\t",self.getStateValuePrediction(root.right.right.data)

    # select the action with the highest Q value
    def selectAction(self, state, explorationRate):
        rand = random.random()
        if rand < explorationRate :
            action = np.random.randint(0, self.nr_actions)
        else :
            action = self.getMaxIndex(self.getPredictedActionValues(state))
        return action

    def selectActionStepsForward(self, state, depth):
        root = Tree()
        # first layer
        predicted1 = self.getPredictedStates(state)
        leftMax = self.getStateValuePrediction(predicted1[0])
        rightMax = self.getStateValuePrediction(predicted1[1])

        predicted2left = self.getPredictedActionValues(predicted1[0])
        leftMax = max(leftMax, np.max(self.getStateValuePrediction(predicted1[0])))

        predicted2right = self.getPredictedStates(predicted1[1])
        rightMax = max(rightMax, np.max(self.getStateValuePrediction(predicted1[1])))

        if rightMax > leftMax:
            return 1
        else:
            return 0


    def addMemory(self, state, action, reward, newState, isFinal):
        self.memory.addMemory(state, action, reward, newState, isFinal)

    def trainStatePredictionOnLastState(self):
        X_batch = np.empty((0,self.state_size), dtype = np.float64)
        Y_batch = np.empty((0,self.state_size), dtype = np.float64)

        lastMemory = self.memory.getLastMemory()
        isFinal = lastMemory['isFinal']
        state = lastMemory['state']
        action = lastMemory['action']
        reward = lastMemory['reward']
        newState = lastMemory['newState']

        X_batch = np.append(X_batch, [state], axis=0)
        Y_batch = np.append(Y_batch, [newState], axis=0)

        self.predictionModels[action].fit(X_batch, Y_batch, batch_size = len(X_batch), verbose = 0)


    def trainStatePreditions(self, miniBatchSize): 
        X_batches = []
        Y_batches = []
        for t in xrange(self.nr_actions):
            X_batches.append(np.empty((0,self.state_size), dtype = np.float64))
            Y_batches.append(np.empty((0,self.state_size), dtype = np.float64))
        miniBatch = self.memory.getMiniBatch(miniBatchSize)
        for sample in miniBatch:
            isFinal = sample['isFinal']
            state = sample['state']
            action = sample['action']
            reward = sample['reward']
            newState = sample['newState']

            inputValues = state.copy()
            targetValues = newState.copy()

            X_batches[action] = np.append(X_batches[action], np.array([inputValues]), axis=0)
            Y_batches[action] = np.append(Y_batches[action], np.array([targetValues]), axis=0)

        for a in xrange(self.nr_actions):
            if len(X_batches[action]) > 0:
                self.predictionModels[action].fit(X_batches[action].reshape(len(X_batches[action]),4), Y_batches[action], batch_size = len(X_batches[action]), verbose = 0)

    def trainRewardModel(self, miniBatchSize): 
        miniBatch = self.memory.getMiniBatch(miniBatchSize)
        X_batch = np.empty((0,self.state_size), dtype = np.float64)
        Y_batch = np.empty((0,1), dtype = np.float64)
        for sample in miniBatch:
            isFinal = sample['isFinal']
            state = sample['state']
            action = sample['action']
            reward = sample['reward']
            newState = sample['newState']

            inputValues = newState.copy()
            targetValue = [self.getTarget(newState, reward, isFinal)]

            X_batch = np.append(X_batch, np.array([inputValues]), axis=0)
            Y_batch = np.append(Y_batch, [targetValue], axis=0)
        self.rewardModel.fit(X_batch, Y_batch, batch_size = len(miniBatch), verbose = 0)
