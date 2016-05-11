# import the gym stuff
import gym
# import other stuff
import random
import numpy as np
# import own classes
from deepq import DeepQ

env = gym.make('CartPole-v0')

epochs = 100000
steps = 100000
explorationRate = 0.75
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
        action = deepQ.selectAction(observation, explorationRate)

        newObservation, reward, done, info = env.step(action)

        # if done:
        #     reward = -50
        deepQ.addMemory(observation, action, reward, newObservation, done)

        deepQ.trainStatePreditions(minibatch_size)
        deepQ.trainRewardModel(minibatch_size)

        observation = newObservation

        if done:
            print "Episode ",epoch," finished after {} timesteps".format(t+1)
            break

    explorationRate -= (2.0/epochs)
    explorationRate = max (0.1, explorationRate)

# env.monitor.close()
# gym.upload('/tmp/wingedsheep-cartpole-democraticDeepQ8', api_key='sk_GC4kfmRSQbyRvE55uTWMOw')