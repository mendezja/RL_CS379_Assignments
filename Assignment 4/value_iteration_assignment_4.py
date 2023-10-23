import gym 
import numpy as np
from collections import Counter, defaultdict
# from tensorboardX import SummaryWriter

ENV_NAME = "FrozenLake-v1"
GAMMA = 0.9 

TEST_EPISODES = 20
SEED = None
SLIP = False
VAL_ITERS = 1

ACTIONS = { 0: 'LEFT ',
            1: 'DOWN ',
            2: 'RIGHT',
            3: 'UP   ' 
            }



class Agent:

    def __init__(self): 
        self.env = self.create_env()

        self.numStates = self.env.observation_space.n
        self.numActions = self.env.action_space.n

        self.value_table = np.zeros(self.numStates)

        self.transits = defaultdict(lambda: Counter())
        self.rewards = defaultdict(lambda: defaultdict(lambda: 0))


    @staticmethod
    def create_env(): 
        return gym.make(ENV_NAME, is_slippery=SLIP) 


    
    def update_transits_rewards(self, state, action, new_state, reward):
        key = (state, action)
        
        self.transits[key][new_state] += 1
        self.rewards[key][new_state] += reward / self.transits[key][new_state] 

        # for key in self.transits:
        #     print(key[0], ACTIONS[key[1]], self.transits[key])

        # for key in self.rewards:
        #     print(key[0], ACTIONS[key[1]], self.rewards[key].items())




    def play_n_random_steps(self, count):

        state = self.env.reset(seed=SEED)


        # For count iterations, step agent w/randomly sampled action
        for i in range(count):

            randAction = self.env.action_space.sample()

            new_state, reward, terminal, _ = self.env.step(randAction)

            # print("\naction: ",ACTIONS[randAction], " state: ", state, " new state: ", new_state, " Reward: ", reward)
 
            # update transits & rewards
            self.update_transits_rewards(state, randAction, new_state, reward)

            # Check if early termination, and reset state
            if terminal: 
                state = self.env.reset(seed=SEED)
            else: 
                state = new_state
    

    def print_value_table(self):

        print("\nState values:")
        val_table = ""

        for i, val in enumerate(self.value_table):
            if i % 4 == 0:
                val_table += ("\n")

            val_table += (" %.4f " % (val))

        print(val_table)

        # for i, val in enumerate(self.value_table):
        #     print("State {:2}: {:.4f}".format(i, val) )
    

    def extract_policy(self):
 
        policy = np.zeros(self.numStates)
        
        # for state in mdp, select action based off policy
        for s in range(self.numStates):  
            policy[s] = self.select_action(s)
            
        return policy



    def print_policy(self, policy):

        print("\nPolicy:")

        str = ""
        for i, val in enumerate(policy):
            if i % 4 == 0:
                str += ("\n")
            str += (" {} ".format (ACTIONS[val]))
        print(str)

        # for i, a in enumerate(self.policy):
        #     print("At state {:2}, move {}".format(i, ACTIONS[a]))



    def calc_action_value(self, state, action):
        
        val = 0 
        # Access all possible transit states
        transit_states = self.transits[(state, action)]

        total_count = sum(transit_states.values())

        # Iterate and use bellman to 
        for s_next in transit_states:             
            prob = transit_states[s_next] / total_count
            reward = self.rewards[(state,action)][s_next]
    
            val += prob * (reward + GAMMA * self.value_table[s_next]) 
        
        return val



    def select_action(self, state):
        
        action_values = np.zeros(self.numActions)
            
        # for a in A(s), calculate the action val
        for a in range(self.numActions):

            action_values[a] = self.calc_action_value(state,a)      
        
        # return best action
        return np.argmax(action_values)





    def play_episode(self, env): 
        # define reward & state
        reward = 0
        state = env.reset(seed=SEED)

        # Loop episode until terminal reached
        terminal = False

        while not terminal: 

            # Select action using policy
            action = self.select_action(state)

            state, r_t, terminal, _ = env.step(action)
        
            # update reward
            reward +=  r_t 

        return reward
            


    def value_iteration(self): 
        
        for _ in range(VAL_ITERS):
            # Calc Bellman Update for all states
            for s in range(self.numStates):
                    
                # Record old state val for Delta
                v = self.value_table[s]                 
                
                # Find max action-value, assign to current state val
                self.value_table[s] =  max([self.calc_action_value(s, a) for a in range(self.numActions)])




if __name__ == "__main__":
    
    
    test_env = Agent.create_env() 

    agent = Agent()  

    iter_no = 0 
    best_reward = float('-inf')

    while True:

        iter_no += 1
        agent.play_n_random_steps(1000)

        agent.value_iteration()

        reward = sum([agent.play_episode(test_env) for _ in range(TEST_EPISODES)]) / TEST_EPISODES


        if reward > best_reward:
            best_reward = reward
            print("Best reward updated %.3f -> %.3f" % (best_reward, reward))

        if reward > 0.80:
            print("Solved in %d iterations!" % iter_no)
            agent.print_value_table()
            policy = agent.extract_policy()
            agent.print_policy(policy)
            break


