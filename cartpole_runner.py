# import the gym stuff
import gym
# import other stuff
import random
import numpy as np
# import own classes
from deepq import DeepQ

env = gym.make('CartPole-v0')

epochs = 500
steps = 10000
explorationRate = 0.75
minibatch_size = 32

deepQ = DeepQ(env, 4)
deepQ.initNetwork([50, 50, 50, 50])

env.monitor.start('/tmp/wingedsheep-cartpole-democraticDeepQ8')
# number of reruns
for epoch in xrange(epochs):
    observation = env.reset()
    print explorationRate
    # number of timesteps
    for t in xrange(steps):
        # env.render()
        qValues = deepQ.getQValues(observation, 0)
        qValues2 = deepQ.getQValues(observation, 1)
        qValues3 = deepQ.getQValues(observation, 2)

        # action = deepQ.selectActionAdded(qValues, qValues2, explorationRate)
        action = deepQ.selectActionMostPreferred(qValues, qValues2, qValues3, explorationRate)
        # action = deepQ.selectActionByProbability(qValues, 1)

        newObservation, reward, done, info = env.step(action)

        # if done:
        #     reward = -50
        deepQ.addMemory(observation, action, reward, newObservation, done)

        deepQ.learnOnMiniBatch(minibatch_size, 0)
        deepQ.learnOnMiniBatch(minibatch_size, 1)
        deepQ.learnOnMiniBatch(minibatch_size, 2)

        observation = newObservation

        if done:
            print "Episode ",epoch," finished after {} timesteps".format(t+1)
            break

    explorationRate -= (2.0/epochs)
    explorationRate = max (0.1, explorationRate)

env.monitor.close()
gym.upload('/tmp/wingedsheep-cartpole-democraticDeepQ8', api_key='sk_GC4kfmRSQbyRvE55uTWMOw')