#!/usr/bin/env python3
import gym
import random
import numpy as np
from collections import Counter, defaultdict
from tensorboardX import SummaryWriter

ENV_NAME = "FrozenLake-v1"
GAMMA = 0.9
EPSILON = 0.9
ALPHA = 0.2
TEST_EPISODES = 20

SEED = None
SLIP = True

ACTIONS = { 0: 'LEFT ',
            1: 'DOWN ',
            2: 'RIGHT',
            3: 'UP   ' 
            }

class Agent:

    def __init__(self):
        # Initialize the environment using gym.make with ENV_NAME 
        self.env = gym.make(ENV_NAME, is_slippery=SLIP) 

        # Set the initial state by resetting the environment
        self.state = self.env.reset(seed=SEED)

        # Capture state and action set sizes (given they are not state dependent)
        self.numStates = self.env.observation_space.n
        self.numActions = self.env.action_space.n

        # Initialize a default dictionary named values for storing the Q-values
        self.q_values = np.zeros([self.numStates, self.numActions])

        # Print variables
        print("Enviroment: ", ENV_NAME )
        print("Gamma: {} | Espilon: {} | Alpha: {} | is_slippery: {} | seed: {}".format(GAMMA,EPSILON,ALPHA,SLIP,SEED))

    def sample_env(self):
        # action = None
        # Select action using epsilon greedy exploration
        if random.random() <= EPSILON:
            action = self.env.action_space.sample()
        else: 
            _, action = self.best_value_and_action(self.state)

        # Use the sampled action to take a step in the environment
        new_state, reward, terminal, _ = self.env.step(action)

        # If the episode ends, reset the environment and store the new state
        old_state = self.state
        self.state = self.env.reset(seed=SEED) if terminal else new_state 
        
        # Return a tuple containing the old state, action, reward, and new state
        return (old_state, action, reward, new_state)


    def best_value_and_action(self, state):

        best_a = np.argmax(self.q_values[state])

        # if all equal, return random action
        if len(set(self.q_values[state])) ==1: 
            best_a = self.env.action_space.sample()

        return (self.q_values[state][best_a], best_a)

        
        

    def value_update(self, state, action, reward, new_state):
        # Call the best_value_and_action function to get the best Q-value for the new state
        best_val, best_action  = self.best_value_and_action(new_state)

        # Calculate the new Q-value using the reward, gamma, and best Q-value of the new state
        new_Q_val = reward + GAMMA * best_val #self.q_values[state][ best_action]
        
        # Update the Q-value of the current state-action pair using alpha and the new Q-value
        self.q_values[state][ action] +=  ALPHA * (new_Q_val - self.q_values[state] [action])

    def play_episode(self, env):
        # Initialize a variable total_reward to 0.0
        total_reward = 0.0
    
        # Reset the environment and store the initial state
        state = env.reset(seed=SEED)

        # Enter a loop that continues until the episode ends
        terminal = False
        while not terminal: 

            # Call the best_value_and_action function to get the best action for the current state
            _, best_action = self.best_value_and_action(state)
            # Take a step in the environment using the best action and store the new state, reward, and done flag
            new_state, reward, terminal, _ = env.step(best_action)

            # Update total_reward using the received reward
            total_reward += reward

            # Update the state using the new state
            state = new_state
            
        # Return the total reward
        return total_reward

    def print_values(self):
        # Print the Q-values in a readable format 
        print ("Q Values")
        # for s in range(self.numStates): 
        #     for a in range(self.numActions):
        #         print("(State: {}, Action: {}) -> {}".format(s, ACTIONS[a], self.q_values[s][a]))

        str = "|   |"
        for a in range(self.numActions):
            str += (" {} |".format (ACTIONS[a]))
        for s in range(self.numStates):
            # if s % 4 == 0:
            str += ("\n| {} |".format(s)) 
            for a in range(self.numActions):
                str += (" %.3f |"% (self.q_values[s][a]))
        print(str)

        


    def print_policy(self):
        # Initialize an empty dictionary named policy
        policy = defaultdict(lambda: "") 
        print("\nPolicy:")

        # # Iterate over all possible states in the environment
        # for s in range(self.numStates): 
        #     # Call the best_value_and_action function to get the best action for each state
        #     _, action = self.best_value_and_action(s)

        #     # Update the policy dictionary with the state-action pair
        #     policy[s] = action

        #     # Print the state and corresponding best action
        #     print("State: {} -> Action: {}".format(s, ACTIONS[action]))
        

        str = ""
        for s in range(self.numStates):
            if s % 4 == 0:
                str += ("\n|")
            _, action = self.best_value_and_action(s)
            str += (" {} |".format (ACTIONS[action]))
        print(str)

        # Return the policy dictionary
        return policy


if __name__ == "__main__":
    test_env = gym.make(ENV_NAME, is_slippery=SLIP)
    agent = Agent()
    writer = SummaryWriter(comment="-q-learning")

    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1
        state, action, reward, new_state = agent.sample_env()
        agent.value_update(state, action, reward, new_state)

        cumulative_reward = 0.0
        for _ in range(TEST_EPISODES):
            cumulative_reward += agent.play_episode(test_env)
        cumulative_reward /= TEST_EPISODES
        writer.add_scalar("reward", cumulative_reward, iter_no)
        if cumulative_reward > best_reward:
            print("Best reward updated %.3f -> %.3f" % (best_reward, cumulative_reward))
            best_reward = cumulative_reward
        if cumulative_reward > 0.80:
            print("Solved in %d iterations!" % iter_no)
            break
    writer.close()

    # Print the Q-values and extract/print the policy
    agent.print_values()
    agent.print_policy()
