import gym
from time import sleep
env = gym.make('Humanoid-v1')
import numpy as np
# import matplotlib.pyplot as plt
from bokeh.plotting import figure, output_file, show


# print('\n\nACTION SPACE')
# print(env.action_space)
# print(env.action_space.low)
# print(env.action_space.high)
# print('\n\nOBSERVATION SPACE')
# print(env.observation_space)
# print(env.observation_space.low)
# print(env.observation_space.high)
# # exit()




from Autoencoder import AutoEncoder as AE

new_ae = AE(in_size=376, downsample=5, lr=0.000001)

states_array = []

energies = []

for i_episode in range(10000):
  observation = env.reset()
  reward = 0
  for t in range(1000):
    # if (t % 10 == 0):
    # env.render()
    # sleep(0.05)
    
    # print(env.observation_space.low)
    # print(reward)
    # print(len(observation))
    # print('len of observation: {}'.format(len(observation)))
    # print(env.observation_space)
    states_array.append(observation)

    
    # if len(states_array) != 20:
    #   states_array.append(observation)
    # else:
    #   np_sa = np.asarray(states_array)
    #   energy = new_ae.train(np_sa)
    #   # print(energy)
    #   energies.append(energy)
    #   print(energies)
    #   states_array = []

    # if t == 20:
    #   print('\n\n\n\n\n\n\n\n\n\n')
    #   print(observation)
    #   print(new_ae.get_reconstruction(np.asarray([observation]))[0])

    action = env.action_space.sample()
    # print(action)
    # print('num actions: {}'.format(len(action)))
    observation, reward, done, info = env.step(action)
    if done:
      np_sa = np.asarray(states_array)
      energy = new_ae.train(np_sa)
      energies.append(energy)
      if i_episode % 250 == 0:
        # print(energies)
        print(energies)
        p = figure(plot_width=400, plot_height=400)
        # arr, energies = zip(enumerate(energies))
        p.line(range(len(energies)), energies, line_width=2)
        show(p)
        # fig, ax = plt.subplots()
        # ax.plot(energies)
        # plt.show()
        # sleep(3.0)
      states_array = []
      print('Episode {} finished after {} timesteps'.format(i_episode, t+1))
      break


