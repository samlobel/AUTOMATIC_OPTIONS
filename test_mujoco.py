import gym
from time import sleep
env = gym.make('Humanoid-v1')

# print('\n\nACTION SPACE')
# print(env.action_space)
# print(env.action_space.low)
# print(env.action_space.high)
# print('\n\nOBSERVATION SPACE')
# print(env.observation_space)
# print(env.observation_space.low)
# print(env.observation_space.high)
# exit()

for i_episode in range(10):
  observation = env.reset()
  reward = 0
  for t in range(1000):
    # if (t % 10 == 0):
    env.render()
    # sleep(0.05)
    
    # print(env.observation_space.low)
    # print(reward)
    # print(observation)
    print('len of observation: {}'.format(len(observation)))
    # print(env.observation_space)
    action = env.action_space.sample()
    # print(action)
    print('num actions: {}'.format(len(action)))
    observation, reward, done, info = env.step(action)
    if done:
      print('Episode finished after {} timesteps'.format(t+1))
      break


