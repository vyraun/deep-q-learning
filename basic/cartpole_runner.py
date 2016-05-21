# import the gym stuff
import gym
# import other stuff
import random
import numpy as np
# import own classes
from deepq import DeepQ

env = gym.make('CartPole-v0')

epochs = 2000
steps = 100000
explorationRate = 1
minibatch_size = 128

deepQ = DeepQ(env)
deepQ.initNetwork([8, 6, 4])

stepCounter = 0
startLearning = 500

# env.monitor.start('/tmp/wingedsheep-cartpole-deepQ11')
# number of reruns
for epoch in xrange(epochs):
    observation = env.reset()
    print explorationRate
    # number of timesteps
    for t in xrange(steps):
        # env.render()
        qValues = deepQ.getQValues(observation)

        action = deepQ.selectAction(qValues, explorationRate)

        newObservation, reward, done, info = env.step(action)

        # if done:
        #     reward = -50
        
        # if done:
        #     deepQ.addMemoryFinal(observation, action, reward, newObservation, done)
        # else:
        deepQ.addMemory(observation, action, reward, newObservation, done)

        if stepCounter >= startLearning:
            deepQ.learnOnMiniBatch(minibatch_size)

        observation = newObservation

        stepCounter += 1

        if done:
            print "Episode ",epoch," finished after {} timesteps".format(t+1)
            break

    # explorationRate -= (2.0/epochs)
    explorationRate *= 0.995
    explorationRate = max (0.05, explorationRate)

deepQ.printNetwork()

# env.monitor.close()
# gym.upload('/tmp/wingedsheep-cartpole-deepQ11', api_key='sk_GC4kfmRSQbyRvE55uTWMOw')