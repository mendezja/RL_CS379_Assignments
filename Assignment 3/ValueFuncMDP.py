# Author: Armando Mendez
# CSCI 397 Building the Value Function

import random
import numpy as np


# State Transition Matrix
# Last s ommited given no possible s'
STM =  [[0.0, 0.6, 0.1, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] ,
        [0.0, 0.0, 0.0, 0.0, 0.3, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0] ,
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.8, 0.0, 0.0, 0.0, 0.0] ,
        [0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.4, 0.4, 0.0, 0.0, 0.0] ,
        [0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0] ,
        [0.4, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3] ,
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.2, 0.0] ,
        [0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.3, 0.0] ,
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0] ,
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5] ,
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

# States labeled by index
# key: 15 indicates terminal state, 0 indicates inital state, 
# 6/11 island rewars -1, 3/11 Reward +2
STATES = [0, -1, -1, 2, 2, -1, 2, -1, -1, -1, 15]

GAMMA = 0.9

ACTIONS = ["move"]


# Closed form solution
# Value function determines the optimal path for our MRP
class ValueFunction():
    def __init__(self):

        P = np.array(STM)
        R = np.array(STATES)
        I = np.identity(len(STATES))

        self.V_pi = np.matmul( np.linalg.inv(I- GAMMA * P), R)



class Agent():

    def __init__(self):
        self.loc = 0
        self.timeStep = 0
        self.reward = 0
        self.V_pi = ValueFunction().V_pi
    
    def act(self):

        # Determine action: Dig (0) or Move (1)
        action = 0 

        self.timeStep += 1

        self.updateLocation()
            
        # Update reward
        self.award(STATES[self.loc])
            
        # Return stats after action 
        return (self.timeStep, self.reward, "T" if self.loc == 10 else self.loc, ACTIONS[action])



    def award(self, quantity):
        self.reward += (GAMMA ** self.timeStep) * quantity


    def updateLocation(self):
        current = self.loc

        # generate rand value [0,1]
        rand_val = random.random()
        p = 0 

        for i in range(len(self.V_pi)):

                # # Get probablility for each state transition
                # p += STM[current][i]
                # print(self.V_pi[i])

                # If probability met, update location
            if STM[current][i] != 0 and self.V_pi[i] > p:
                p = self.V_pi[i]
                self.loc = i
                    
        # return
        # print("Error: Agent failed to Move")
    # def updateLocation(self):
    #     current = self.loc

    #     # generate rand value [0,1]
    #     rand_val = random.random()
    #     p = 0 

    #     for i in range(len(STM[0])):

    #             # Get probablility for each state transition
    #             p += STM[current][i]

    #             # If probability met, update location
    #             if STM[current][i] != 0 and rand_val <= p:  
    #                 self.loc = i
    #                 return

    #     print("Error: Agent failed to Move")


def main():

    # Format prints
    stepReport = "{:2d}. Reward: {:.2f} | State: {} | Action: {}"

    # Show optimal path episode
    agent = Agent()

    print("\nEpisode 1, optimal path using Value Function")
    print(stepReport.format(0, 0.00, "I", "null" ))


    # 25 steps or till terminal reached
    while agent.timeStep < 25 and agent.loc !=10:

        data = agent.act() 

        print(stepReport.format(*data)) 

    print("Cumulative Reward: {:.2f}".format(agent.reward))



    


if __name__ == main():
    main()
