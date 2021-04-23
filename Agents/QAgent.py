'''
Q Learning Agent
'''

import random
import numpy as np

class QAgentOptions:
    def __init__(self):
        self.max_epsilon = 1
        self.min_epsilon = 0.001
        self.epsilon_decay = 0.1
        self.gamma = 0.99
        self.learning_rate = 0.01
        self.verbose = True
        self.verbose_freq = 1
        self.render_env = False
        self.save_path = "./q_agent_model"


class QAgent:
    def __init__(self,n_states, n_actions, options = QAgentOptions()):
        self.n_states = n_states
        self.n_actions = n_actions
        self.q_table = np.zeros((n_states, n_actions))
        self.options = options
        self.curr_epsilon = options.max_epsilon
   

    def act(self, state):
        '''
        Acts on greedy epsilon
        '''
        action = 0
        if(random.uniform(0, 1) > (1-self.curr_epsilon)):
            action = random.randint(0, self.n_actions-1)
        else:
            # Greedy action
            action = np.argmax(self.q_table[state])
        return action

    def greedy_act(self, state):
        '''
        Returns the greedy action given a state
        '''
        return np.argmax(self.q_table[state])

    def train(self, n_episodes, n_steps, env):
        '''
        Trains the Q-agent
        n_episodes: Number of episodes
        n_steps: Number of steps
        env: Corresponding gym environment. TODO: Implement a custom interface to act as a wrapper
        '''
        self.curr_epsilon = self.options.max_epsilon
        returns = []
        for e in range(n_episodes):
            curr_state = env.reset()
          

            # Decay epsilon
            self.curr_epsilon = self.options.min_epsilon + (self.options.max_epsilon - self.options.min_epsilon)*np.exp(-self.options.epsilon_decay*e)
            total_reward = 0
            for s in range(n_steps):
                if self.options.render_env:
                    env.render()
                # Act first
                action = self.act(curr_state)
                next_state, reward, done, info = env.step(action)
                total_reward += reward
                # Update Q table
                max_act_val = np.max(self.q_table[next_state])
                self.q_table[curr_state][action] = self.q_table[curr_state][action] + self.options.learning_rate*(reward + max_act_val*self.options.gamma -  self.q_table[curr_state][action])

                if done:
                    break
                curr_state = next_state
            print(total_reward)
            returns.append(total_reward)
            if e > 0 and e%self.options.verbose_freq == 0 and self.options.verbose:
                print("Episode: {} - Return: {}".format(e, returns[-1]))
        
        self.save(self.options.save_path)

    def test(self, n_episodes, n_steps, env):
        returns = []
        for e in range(n_episodes):
            curr_state = env.reset()
            total_rewards = 0
            print("Episode:", e)
            for s in range(n_steps):
                env.render()
                action = self.greedy_act(curr_state)
                next_state, reward, done, info = env.step(action)
                total_rewards += reward
                
                if done or s == n_steps-1:
                    rewards.append(total_rewards)
                    break
                curr_state = next_state

        print ("Score over time: " +  str(sum(returns)/n_episodes))

    def save(self, filepath):
        np.save(filepath, self.q_table)
  

    def load(self, filepath):
        self.q_table = np.load(filepath)

            
