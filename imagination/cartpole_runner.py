# import the gym stuff
import gym
# import other stuff
import random
import numpy as np
# import own classes
from deepq import DeepQ

env = gym.make('CartPole-v0')

epochs = 1000
steps = 100000
explorationRate = 0
minibatch_size = 32

deepQ = DeepQ(env, 4)
deepQ.initImaginationNetworks()
deepQ.initRewardNetwork()

# env.monitor.start('/tmp/wingedsheep-cartpole-democraticDeepQ8')
# number of reruns
for epoch in xrange(epochs):
    observation = env.reset()
    print explorationRate
    # number of timesteps
    for t in xrange(steps):
        env.render()
        # action = deepQ.selectAction(observation, explorationRate)
        action = deepQ.selectActionStepsForward(observation, 2)

        # print action

        # predictedStateLeft = deepQ.getStatePrediction(observation, 0)
        # predictedStateRight = deepQ.getStatePrediction(observation, 1)
        # predictedActionValues = deepQ.getPredictedActionValues(observation)
        # deepQ.printStateValueTree(observation)
        # print action

        # print("left: ",predictedStateLeft[0],predictedStateLeft[1],predictedStateLeft[2],predictedStateLeft[3] )
        # print("right: ",predictedStateRight[0],predictedStateRight[1],predictedStateRight[2],predictedStateRight[3] )
        # predictedStateLeft[0] *= 2
        # predictedStateRight[0] *= 2
        # predictedStateLeft[1] *= 2
        # predictedStateRight[1] *= 2
        # predictedStateLeft[2] *= 3
        # predictedStateRight[2] *= 3
        # predictedStateLeft[3] *= 3
        # predictedStateRight[3] *= 3

        # errorLeft = sum(map((lambda x: x **2),predictedStateLeft))
        # errorRight = sum(map((lambda x: x **2),predictedStateRight))
        # if (errorLeft > errorRight):
        #     action = 1
        # else :
        #     action = 0

        # rand = random.random()
        # if rand < explorationRate :
        #     action = np.random.randint(0, 2)

        # if action == 0:
        # 	predictedState = predictedStateLeft
        # else :
        # 	predictedState = predictedStateRight

        # print action

        # print "old state: ",predictedState

        newObservation, reward, done, info = env.step(action)

        # print "new state: ",newObservation
        # difference = newObservation - predictedState
        # error = sum(map((lambda x: x **2),difference))
        # print "difference: ",error
        # print "error: ",error

        # if done:
        #     reward = -50
        deepQ.addMemory(observation, action, reward, newObservation, done)

        # deepQ.trainStatePredictionOnLastState()
        deepQ.trainStatePreditions(minibatch_size)
        deepQ.trainRewardModel(minibatch_size)

        observation = newObservation

        if done:
            print "Episode ",epoch," finished after {} timesteps".format(t+1)
            break

    explorationRate -= (2.0/epochs)
    explorationRate = max (0.05, explorationRate)

# env.monitor.close()
# gym.upload('/tmp/wingedsheep-cartpole-democraticDeepQ8', api_key='sk_GC4kfmRSQbyRvE55uTWMOw')