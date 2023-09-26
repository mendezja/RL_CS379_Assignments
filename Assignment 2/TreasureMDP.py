# Author: Armando Mendez
# CSCI 397 Treasure MDP

# Questions: 
# - Reward: -1 | State: 0 | Action: Moved how? 
# - should the time step increase for both big and move
# - how to implement gamma

import random

STM =  [[0.0, 0.6, 0.1, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] ,
        [0.0, 0.0, 0.0, 0.0, 0.3, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0] ,
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.8, 0.0, 0.0, 0.0, 0.0] ,
        [0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.4, 0.4, 0.0, 0.0, 0.0] ,
        [0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0] ,
        [0.4, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3] ,
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.2, 0.0] ,
        [0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.3, 0.0] ,
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0] ,
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5]]

STATES = [0,0,0,2,2,0,2,0,0,0,5]

class Agent():

    gamma = 0.9
    
    def __init__(self):
        self.loc = 0
        self.timeStep = 0
        self.reward = 0
        self.t_found = 0
    
    def act(self):
        
        moved = True 
        self.timeStep += 1

        # Determine action: Dig or Move
        if random.random() <= 0.1:

            # Update Moved to False
            moved = False

            # if dig, check for treasure and update attributes
            if STATES[self.loc] == 2 :
                self.t_found += 1
                self.award(2) 
            
                STATES[self.loc] = 0
            
        else:
            # Update reward
            self.award(-1)

            rand_val = random.random()
            p = 0

            # Calculate new loc given current loc 
            for i in range(len(STM[0])):

                # Get probablility for each state transition
                p += STM[self.loc][i]

                # If prob met, update location
                if STM[self.loc][i] != 0 and rand_val <= p:  
                    self.loc = i
                    break
            
            # Check if terminal state
            if STATES[self.loc] == 5: 

                # Update Reward accordening
                self.award( 20 if self.t_found == 3 else 5)

                # Return Episode total Reward
                return "Terminal Reached, Reward: {:.2f}".format(self.reward)

        # Return stats after action
        return "Reward: {:.2f} | State: {} | Action: {}".format(self.reward, self.loc, "Moved" if moved else "Dug")
        
    def award(self, quantity):
        self.reward += (self.gamma ** self.timeStep) * quantity


def main():
    # history store
    history = [[] for _ in range(10)]

    for i in range(10):

        agent = Agent()
        print()
        print("Episode ", i + 1)
        print("Reward:  0.00 | State: 0 | Action: None")

        while agent.timeStep < 25 and agent.loc !=10:
            action = agent.act()
            print(action)
            history[i].append(action) 
    


if __name__ == main():
    main()
