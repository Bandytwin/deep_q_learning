import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import random

from collections import deque
import gym

# gym environment to be solved
ENV_ID = "CartPole-v1"

# learning rate for the adam optimizer of NN
ADAM_ALPHA = 0.001
# discount factor to apply to all future rewards
GAMMA = 0.95

# batch size to use when training the NN
BATCH_SIZE = 64

# parameters controlling the amount of epsilon-greedy exploration
EPSILON_MAX = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.990

class deepAgent:
    def __init__(self, action_space, state_space):
        self.obs_dim = state_space.shape[0]
        self.action_dim = action_space.n
        self.epsilon = EPSILON_MAX
        self.memory = deque(maxlen=2000)
        self.init_model()

    def mem_reset(self):
        '''
        Reset the variable that stores relevant state, action, and return values
        :return: None
        '''
        self.memory = deque(maxlen=2000)

    def init_model(self):
        '''
        Define and compile the neural network
        :return: None
        '''
        self.model = Sequential()
        self.model.add(Dense(12, input_dim=self.obs_dim, activation="relu"))
        self.model.add(Dense(12, activation="relu"))
        self.model.add(Dense(self.action_dim, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=ADAM_ALPHA))

    def build_history(self, action, state, next_state, reward, done):
        '''
        Store data related to a new iteration of state-action-next_state
        :param action: Action taken by agent
        :param state: The current environment state
        :param next_state: The resulting state after the action is taken
        :param reward: Immediate reward
        :param done: Indicates if the current episode has ended
        :return: None
        '''
        state = np.reshape(state, [1, 4])
        next_state = np.reshape(next_state, [1, 4])
        self.memory.append((state, action, reward, next_state, done))

    def calc_actual_returns(self):
        '''
        Loop through everything stored in memory, and calculate the actual return of each state-action
        :return: A numpy array of the actual discounted return for all samples stored in memory
        '''
        actual_returns = []
        sample_reward = []
        sample_return = []
        for state, action, reward, next_state, done in self.memory:
            sample_reward.append(reward)
            if done is False:
                continue
            else:
                sample_return.append(sample_reward[-1])
                for ind in range(len(sample_reward) - 1):
                    reward_ind = -(ind + 2)
                    next_return = round(sample_reward[reward_ind] + GAMMA*sample_return[ind], ndigits=3)
                    sample_return.append(next_return)
                sample_return.reverse()
                actual_returns = actual_returns + sample_return
                sample_return = []
                sample_reward = []
        return np.array(actual_returns)

    def calc_estimated_returns(self):
        '''
        Loop through everything stored in memory, and calculate the predicted return of each state-action based on the
        current model.
        :return: A numpy array of the predicted discounted return for all samples stored in memory
        '''
        states = [item[0] for item in self.memory]
        actions = [item[1] for item in self.memory]
        states = np.reshape(states, [len(self.memory), 4])
        est_returns = self.model.predict(states)
        est_returns = [value[action] for value, action in zip(est_returns, actions)]
        return np.array(est_returns)

    def update_policy(self):
        '''
        Perform a policy update
        :return: None
        '''
        if len(self.memory) < BATCH_SIZE:
            return
        memory_batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, next_state, done in memory_batch:
            old_state_q = self.model.predict(state)
            q_update = reward
            if done is False:
                q_update = reward + GAMMA*np.amax(self.model.predict(next_state))
            old_state_q[0, action] = q_update
            self.model.fit(state, old_state_q, verbose=0, epochs=1)
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY

    def select_action(self, current_state):
        '''
        Choose the next action based on the current state
        :param current_state: The current state that the agent is in
        :return: The highest value action based on the current model (with probability 1 - epsilon) otherwise a random
        action
        '''
        if np.random.uniform() < self.epsilon:
            return np.random.choice([0, 1])
        else:
            best_action = np.argmax(self.model.predict(current_state))
            return best_action

def run_simulation(num_episodes):
    # test deep-Q learning
    env = gym.make(ENV_ID)
    agent = deepAgent(env.action_space, env.observation_space)
    agent.init_model()
    perf_hist = []
    mse_list = []
    agent.mem_reset()

    for i_episode in range(num_episodes):
        observation = np.reshape(env.reset(), [1, 4])
        for t in range(1000):
            #env.render()
            action = agent.select_action(observation)
            new_observation, reward, done, info = env.step(action)
            new_observation = np.reshape(new_observation, [1, 4])
            agent.build_history(action, observation, new_observation, reward, done)
            observation = new_observation
            if done:
                break
        perf_hist.append(t)

        # periodically calculate action-value model accuracy
        if i_episode % 20 == 0:
            actual_returns = agent.calc_actual_returns()
            predicted_returns = agent.calc_estimated_returns()
            mse = ((actual_returns - predicted_returns) ** 2).mean(axis=0)
            mse_list.append(mse)

        print("episode: {}/{}, score: {}"
              .format(i_episode, num_episodes, t+1))
        agent.update_policy()
    env.close()
    return perf_hist, mse_list, agent

if __name__ == "__main__":
    # perform a full training simulation
    perf_hist, mse_list, trained_agent = run_simulation(1000)
    plt.plot(mse_list)
    plt.plot(perf_hist)

    # plot  MSE and performance over time
    mse_x = [i * 5 for i in list(range(len(mse_list)))]

    fig, axs = plt.subplots(2, 1)
    axs[0].plot(mse_x, mse_list)
    #axs[0].set_xlim(0, 2)
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('MSE')
    axs[0].grid(True)

    axs[1].plot(list(range(len(perf_hist))), perf_hist)
    axs[1].set_ylabel('Epsiode Return')
    axs[1].set_xlabel('Episode')

    fig.tight_layout()
    plt.show()





