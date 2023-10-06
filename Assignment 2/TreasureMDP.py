# Author: Armando Mendez
# CSCI 397 Treasure MDP

import random


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
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5]]

# States labeled by index
# key: 0 indicates no treaure, 2 indicates treaure, 5 indicates terminal state
STATES = [0,0,0,2,2,0,2,0,0,0,5]

GAMMA = 0.9

ACTIONS = ["dig", "move"]


class Agent():

    def __init__(self):
        self.loc = 0
        self.timeStep = 0
        self.reward = 0
        self.t_found = 0
    
    def act(self):

        # Determine action: Dig (0) or Move (1)
        action = 0 if random.random() <= 0.1 else 1

        self.timeStep += 1

        # Dig
        if action == 0:

            # Check for treasure and update attributes
            if STATES[self.loc] == 2 :
                self.t_found += 1
                self.award(2) 
            
                STATES[self.loc] = 0

        # Move
        else:
            # Update reward
            self.award(-1)

            # Update Location
            self.updateLocation()
            
            # Check if terminal state
            if STATES[self.loc] == 5: 
                self.award( 20 if self.t_found == 3 else 5)

        # Return stats after action 
        return (self.timeStep, self.reward, "T" if self.loc == 10 else self.loc, ACTIONS[action])



    def award(self, quantity):
        self.reward += (GAMMA ** self.timeStep) * quantity


    def updateLocation(self):
        current = self.loc

        # generate rand value [0,1]
        rand_val = random.random()
        p = 0 

        for i in range(len(STM[0])):

                # Get probablility for each state transition
                p += STM[current][i]

                # If probability met, update location
                if STM[current][i] != 0 and rand_val <= p:  
                    self.loc = i
                    return

        print("Error: Agent failed to Move")


def main():

    # Store episodes data
    history = [[] for _ in range(10)]

    stepReport = "{:2d}. Reward: {:.2f} | State: {} | Action: {}"

    # 10 episodes
    for i in range(10):

        agent = Agent()

        print("\nEpisode ", i + 1)
        print(stepReport.format(0, 0.00, "I", "null" ))


        # 25 steps or till terminal reached
        while agent.timeStep < 25 and agent.loc !=10:

            data = agent.act()
            history[i].append(data) 

            print(stepReport.format(*data))

        print("Cumulative Reward: {:.2f}".format(agent.reward))
    


if __name__ == main():
    main()
