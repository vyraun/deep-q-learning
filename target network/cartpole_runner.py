# import the gym stuff
import gym
# import other stuff
import random
import numpy as np
# import own classes
from deepq import DeepQ

env = gym.make('CartPole-v0')

epochs = 10000
steps = 100000
updateTargetNetwork = 10000
explorationRate = 1
minibatch_size = 32
learnStart = 32
learningRate = 0.00025
discountFactor = 0.99
memorySize = 1000000

deepQ = DeepQ(4, 2, memorySize, discountFactor, learningRate, learnStart)
deepQ.initNetworks([24, 16, 12, 8])

stepCounter = 0

# env.monitor.start('/tmp/wingedsheep-cartpole-democraticDeepQ8')
# number of reruns
for epoch in xrange(epochs):
    observation = env.reset()
    print explorationRate
    # number of timesteps
    for t in xrange(steps):
        env.render()
        qValues = deepQ.getQValues(observation)

        action = deepQ.selectAction(qValues, explorationRate)

        newObservation, reward, done, info = env.step(action)

        # if done:
        #     reward = -50
        deepQ.addMemory(observation, action, reward, newObservation, done)

        if stepCounter >= learnStart:
            deepQ.learnOnMiniBatch(minibatch_size)

        observation = newObservation

        if done:
            print "Episode ",epoch," finished after {} timesteps".format(t+1)
            break

        stepCounter += 1
        if stepCounter % updateTargetNetwork == 0:
            deepQ.updateTargetNetwork()
            print "updating target network"

    explorationRate *= 0.995
    # explorationRate -= (2.0/epochs)
    explorationRate = max (0.05, explorationRate)

deepQ.printNetwork()

# env.monitor.close()
# gym.upload('/tmp/wingedsheep-cartpole-democraticDeepQ8', api_key='sk_GC4kfmRSQbyRvE55uTWMOw')