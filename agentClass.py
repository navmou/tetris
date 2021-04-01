import numpy as np
import random
import math
import h5py
import hdfdict

# This file provides the skeleton structure for the classes TQAgent and TDQNAgent to be completed by you, the student.
# Locations starting with # TO BE COMPLETED BY STUDENT indicates missing code that should be written by you.

class TQAgent:
    # Agent for learning to play tetris using Q-learning
    def __init__(self,alpha,epsilon,episode_count):
        # Initialize training parameters
        self.alpha=alpha
        self.epsilon=epsilon
        self.episode=0
        self.episode_count=episode_count
        self.reward_tots = [0]
        self.saving = []
        self.game_reward_sum = 0
        self.game_reward_sum_list = []
        self.moving_average = []
    def fn_init(self,gameboard):
        self.gameboard=gameboard
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # In this function you could set up and initialize the states, actions and Q-table and storage for the rewards
        # This function should not return a value, store Q table etc as attributes of self

        # Useful variables: 
        # 'gameboard.N_row' number of rows in gameboard
        # 'gameboard.N_col' number of columns in gameboard
        # 'len(gameboard.tiles)' number of different tiles
        # 'self.episode_count' the total number of episodes in the training
        
        # actions are different for each block, so the block itself is a part of state
        # block1: 2*2 square , block2: angle , block3: diagonal , block4: line
        # for block1 there are 3 horizontal moves x = [-2 , -1 , 0]
        # for block2 there are 3 horizontal moves and 4 rotations. then there are 12 moves
        # for block3 there are 3 horizontal moves and 2 rotations. then there are 6 moves
        # for block4 there are 4 horizontal moves for vertical orientation and 3 horizontal moves for horizontal orientation
        # which will give a 7 moves in total.
        # The other part of state can be the height of the columns.
        # Then the total state can be represented by an array of 5 entries, first element can be the block representer
        # the remainig four will be the hieght of each column.
        self.Q0 = {(0,0,0,0,0):[0]*7}
        self.Q1 = {(1,0,0,0,0):[0]*6}
        self.Q2 = {(2,0,0,0,0):[0]*12}
        self.Q3 = {(3,0,0,0,0):[0]*3}
        
        
        
        
    
        
        

    def fn_load_strategy(self,strategy_file):
        # TO BE COMPLETED BY STUDENT
        # Here you can load the Q-table (to Q-table of self) from the input parameter strategy_file (used to test how the agent plays)
        
        self.Q0 = self.change_to_int(dict(hdfdict.load(f'{strategy_file}0.h5')))
        self.Q1 = self.change_to_int(dict(hdfdict.load(f'{strategy_file}1.h5')))
        self.Q2 = self.change_to_int(dict(hdfdict.load(f'{strategy_file}2.h5')))
        self.Q3 = self.change_to_int(dict(hdfdict.load(f'{strategy_file}3.h5')))
        
        
    
    def change_to_int(self, Q):
        transformed = {}
        for key in Q.keys():
            new_key = key.split(sep="_")
            int_key = tuple([int(i) for i in new_key])
            transformed[int_key] = Q[key]
        
        return transformed
    
        
        
        
        

    def fn_read_state(self):
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # In this function you could calculate the current state of the game board
        # You can for example represent the state as an integer entry in the Q-table
        # This function should not return a value, store the state as an attribute of self

        # Useful variables: 
        # 'self.gameboard.N_row' number of rows in gameboard
        # 'self.gameboard.N_col' number of columns in gameboard
        # 'self.gameboard.board[index_row,index_col]' table indicating if row 'index_row' and column 'index_col' is occupied (+1) or free (-1)
        # 'self.gameboard.cur_tile_type' identifier of the current tile that should be placed on the game board (integer between 0 and len(self.gameboard.tiles))
        
        dummy_state = [self.gameboard.cur_tile_type , 0 , 0 , 0 , 0]
        for column in range(self.gameboard.N_col):
            for row in range(self.gameboard.N_row):
                if self.gameboard.board[self.gameboard.N_row-1 - row][column] == 1:
                    dummy_state[column+1] = self.gameboard.N_row - row
                    break
            
        self.state = tuple(dummy_state)
        
        if self.state[0] == 0:
            if self.state not in self.Q0.keys():
                self.Q0[self.state] = [0]*7
                
        elif self.state[0] == 1:
            if self.state not in self.Q1.keys():
                self.Q1[self.state] = [0]*6
                
        elif self.state[0] == 2:
            if self.state not in self.Q2.keys():
                self.Q2[self.state] = [0]*12
                
        elif self.state[0] == 3:
            if self.state not in self.Q3.keys():
                self.Q3[self.state] = [0]*3
        
        print(self.state)
                
        
        
        
        
        
        
        
                   
                    
                    
    def fn_execute_action(self):
        if self.gameboard.cur_tile_type == 0:
            if self.action == 0:
                self.gameboard.fn_move(self.gameboard.tile_x , self.gameboard.tile_orientation)
            elif self.action == 1:
                self.gameboard.fn_move(self.gameboard.tile_x+1 , self.gameboard.tile_orientation)
            elif self.action == 2:
                self.gameboard.fn_move(self.gameboard.tile_x-1 , self.gameboard.tile_orientation)
            elif self.action == 3:
                self.gameboard.fn_move(self.gameboard.tile_x-2 , self.gameboard.tile_orientation)
            elif self.action == 4:
                self.gameboard.fn_move(self.gameboard.tile_x , (self.gameboard.tile_orientation+1))
            elif self.action == 5:
                self.gameboard.fn_move(self.gameboard.tile_x-1 , (self.gameboard.tile_orientation+1))
            elif self.action == 6:
                self.gameboard.fn_move(self.gameboard.tile_x-2 , (self.gameboard.tile_orientation+1))
            
        elif self.gameboard.cur_tile_type == 1:
            if self.action == 0:
                self.gameboard.fn_move(self.gameboard.tile_x , self.gameboard.tile_orientation)
            elif self.action == 1:
                self.gameboard.fn_move(self.gameboard.tile_x-1 , self.gameboard.tile_orientation)
            elif self.action == 2:
                self.gameboard.fn_move(self.gameboard.tile_x-2 , self.gameboard.tile_orientation)
            elif self.action == 3:
                self.gameboard.fn_move(self.gameboard.tile_x , (self.gameboard.tile_orientation+1))
            elif self.action == 4:
                self.gameboard.fn_move(self.gameboard.tile_x-1 , (self.gameboard.tile_orientation+1))
            elif self.action == 5:
                self.gameboard.fn_move(self.gameboard.tile_x-2 , (self.gameboard.tile_orientation+1))
        
        elif self.gameboard.cur_tile_type == 2:
            if self.action == 0:
                self.gameboard.fn_move(self.gameboard.tile_x , self.gameboard.tile_orientation)
            elif self.action == 1:
                self.gameboard.fn_move(self.gameboard.tile_x-1 , self.gameboard.tile_orientation)
            elif self.action == 2:
                self.gameboard.fn_move(self.gameboard.tile_x-2 , self.gameboard.tile_orientation)
            elif self.action == 3:
                self.gameboard.fn_move(self.gameboard.tile_x , (self.gameboard.tile_orientation+1))
            elif self.action == 4:
                self.gameboard.fn_move(self.gameboard.tile_x-1 , (self.gameboard.tile_orientation+1))
            elif self.action == 5:
                self.gameboard.fn_move(self.gameboard.tile_x-2 , (self.gameboard.tile_orientation+1))
            elif self.action == 6:
                self.gameboard.fn_move(self.gameboard.tile_x , (self.gameboard.tile_orientation+2))
            elif self.action == 7:
                self.gameboard.fn_move(self.gameboard.tile_x-1 , (self.gameboard.tile_orientation+2))
            elif self.action == 8:
                self.gameboard.fn_move(self.gameboard.tile_x-2 , (self.gameboard.tile_orientation+2))
            elif self.action == 9:
                self.gameboard.fn_move(self.gameboard.tile_x , (self.gameboard.tile_orientation+3))
            elif self.action == 10:
                self.gameboard.fn_move(self.gameboard.tile_x-1 , (self.gameboard.tile_orientation+3))
            elif self.action == 11:
                self.gameboard.fn_move(self.gameboard.tile_x-2 , (self.gameboard.tile_orientation+3))
            
        elif self.gameboard.cur_tile_type == 3:
            if self.action == 0:
                self.gameboard.fn_move(self.gameboard.tile_x , self.gameboard.tile_orientation)
            elif self.action == 1:
                self.gameboard.fn_move(self.gameboard.tile_x-1 , self.gameboard.tile_orientation)
            elif self.action == 2:
                self.gameboard.fn_move(self.gameboard.tile_x-2 , self.gameboard.tile_orientation)        


    def fn_select_action(self):
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # Choose and execute an action, based on the Q-table or random if epsilon greedy
        # This function should not return a value, store the action as an attribute of self and exectute the action by moving the tile to the desired position and orientation

        # Useful variables: 
        # 'self.epsilon' parameter epsilon in epsilon-greedy policy

        # Useful functions
        # 'self.gameboard.fn_move(tile_x,tile_orientation)' use this function to execute the selected action
        # The input argument 'tile_x' contains the column of the tile (0 < tile_x < self.gameboard.N_col)
        # The input argument 'tile_orientation' contains the number of 90 degree rotations of the tile (0 < tile_orientation < # of non-degenerate rotations)
        # The function returns 1 if the action is not valid and 0 otherwise
        # You can use this function to map out which actions are valid or not
        
        if self.epsilon < np.random.uniform():
            if self.state[0] == 0:
                self.action = np.argmax(self.Q0[self.state])
                print(self.Q0[self.state])
            elif self.state[0] == 1:
                self.action = np.argmax(self.Q1[self.state])
                print(self.Q1[self.state])
            elif self.state[0] == 2:
                self.action = np.argmax(self.Q2[self.state])
                print(self.Q2[self.state])
            elif self.state[0] == 3:
                self.action = (np.argmax(self.Q3[self.state]))
                print(self.Q3[self.state])
        
        else:
            if self.state[0] == 0:
                self.action = np.random.choice(range(0,7))
            elif self.state[0] == 1:
                self.action = np.random.choice(range(0,6))
            elif self.state[0] == 2:
                self.action = np.random.choice(range(0,12))
            elif self.state[0] == 3:
                self.action = np.random.choice(range(0,3))
    
        print(f'self.gameboard.tile_x: {self.gameboard.tile_x}')

        self.fn_execute_action()
    
    
    
    
    
    
    
    
    
    
    def fn_reinforce(self,old_state,reward):
        pass
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # Update the Q table using state and action stored as attributes in self and using function arguments for the old state and the reward
        # This function should not return a value, the Q table is stored as an attribute of self

        # Useful variables: 
        # 'self.alpha' learning rate
        if old_state[0] == 0:
            if self.state[0] == 0:
                self.Q0[old_state][self.action] += self.alpha*(reward + np.max(self.Q0[self.state]) - self.Q0[old_state][self.action])
            elif self.state[0] == 1:
                self.Q0[old_state][self.action] += self.alpha*(reward + np.max(self.Q1[self.state]) - self.Q0[old_state][self.action])
            elif self.state[0] == 2:
                self.Q0[old_state][self.action] += self.alpha*(reward + np.max(self.Q2[self.state]) - self.Q0[old_state][self.action])
            elif self.state[0] == 3:
                self.Q0[old_state][self.action] += self.alpha*(reward + np.max(self.Q3[self.state]) - self.Q0[old_state][self.action])
                
        elif old_state[0] == 1:
            if self.state[0] == 0:
                self.Q1[old_state][self.action] += self.alpha*(reward + np.max(self.Q0[self.state]) - self.Q1[old_state][self.action])
            elif self.state[0] == 1:
                self.Q1[old_state][self.action] += self.alpha*(reward + np.max(self.Q1[self.state]) - self.Q1[old_state][self.action])
            elif self.state[0] == 2:
                self.Q1[old_state][self.action] += self.alpha*(reward + np.max(self.Q2[self.state]) - self.Q1[old_state][self.action])
            elif self.state[0] == 3:
                self.Q1[old_state][self.action] += self.alpha*(reward + np.max(self.Q3[self.state]) - self.Q1[old_state][self.action])
        
        elif old_state[0] == 2:
            if self.state[0] == 0:
                self.Q2[old_state][self.action] += self.alpha*(reward + np.max(self.Q0[self.state]) - self.Q2[old_state][self.action])
            elif self.state[0] == 1:
                self.Q2[old_state][self.action] += self.alpha*(reward + np.max(self.Q1[self.state]) - self.Q2[old_state][self.action])
            elif self.state[0] == 2:
                self.Q2[old_state][self.action] += self.alpha*(reward + np.max(self.Q2[self.state]) - self.Q2[old_state][self.action])
            elif self.state[0] == 3:
                self.Q2[old_state][self.action] += self.alpha*(reward + np.max(self.Q3[self.state]) - self.Q2[old_state][self.action])
           
        elif old_state[0] == 3:
            if self.state[0] == 0:
                self.Q3[old_state][self.action] += self.alpha*(reward + np.max(self.Q0[self.state]) - self.Q3[old_state][self.action])
            elif self.state[0] == 1:
                self.Q3[old_state][self.action] += self.alpha*(reward + np.max(self.Q1[self.state]) - self.Q3[old_state][self.action])
            elif self.state[0] == 2:
                self.Q3[old_state][self.action] += self.alpha*(reward + np.max(self.Q2[self.state]) - self.Q3[old_state][self.action])
            elif self.state[0] == 3:
                self.Q3[old_state][self.action] += self.alpha*(reward + np.max(self.Q3[self.state]) - self.Q3[old_state][self.action])
           
        
        
        
        

    def fn_turn(self ):
        if self.gameboard.gameover:
            self.episode+=1
            self.game_reward_sum_list.append(self.game_reward_sum)
            self.reward_tots.append(self.game_reward_sum)
            self.game_reward_sum = 0
            if self.episode%100==0:
                begin = self.episode-100
                end = self.episode
                print(f'episode {(self.episode)}/{(self.episode_count)} (reward: {np.sum(self.reward_tots[begin:end])})')
                self.moving_average.append(np.sum(self.reward_tots[begin:end])/100)
                print(f'episode {self.episode} reward: {self.game_reward_sum_list[self.episode-1]}')
            if self.episode%1000==0:
                saveEpisodes=[0,100,200,500,1000,20000,50000,100000,200000,500000,1000000];
                if self.episode in saveEpisodes:
                    # TO BE COMPLETED BY STUDENT
                    # Here you can save the rewards and the Q-table to data files for plotting of the rewards and the Q-table can be used to test how the agent plays
                    pass
                    
            if self.episode>=self.episode_count:
                hdfdict.dump(self.Q0, "Q0.h5")
                hdfdict.dump(self.Q1, "Q1.h5")
                hdfdict.dump(self.Q2, "Q2.h5")
                hdfdict.dump(self.Q3, "Q3.h5")
                with open('rewards.txt' , 'w') as f:
                    for i in self.reward_tots:
                        f.write(f'{i}\n')
                with open('game_rewards.txt' , 'w') as f:
                    for i in self.game_reward_sum_list:
                        f.write(f'{i}\n')
                with open('moving_avg.txt' , 'w') as f:
                    for i in self.moving_average:
                        f.write(f'{i}\n')
                raise SystemExit(0)
            else:
                self.gameboard.fn_restart()
        else:
            
            self.fn_read_state()
            #print(f'state = {self.state}')
            #print(self.gameboard.board)
            # Select and execute action (move the tile to the desired column and orientation)
            self.fn_select_action()
            #print(f'action = {self.action}')
            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to copy the old state into the variable 'old_state' which is later passed to fn_reinforce()
            old_state = self.state
            #print(f'old_state = {old_state}')
            # Drop the tile on the game board
            reward=self.gameboard.fn_drop()
            self.game_reward_sum += reward
            #print(f'reward = {reward}')
            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to add the current reward to the total reward for the current episode, so you can save it to disk later
            #self.reward_tots.append(reward)
            #print(f'reward_tots = {self.reward_tots}')
            # Read the new state
            self.fn_read_state()
            
            # Update the Q-table using the old state and the reward (the new state and the taken action should be stored as attributes in self)
            self.fn_reinforce(old_state,reward)


class TDQNAgent:
    # Agent for learning to play tetris using Q-learning
    def __init__(self,alpha,epsilon,epsilon_scale,replay_buffer_size,batch_size,sync_target_episode_count,episode_count):
        # Initialize training parameters
        self.alpha=alpha
        self.epsilon=epsilon
        self.epsilon_scale=epsilon_scale
        self.replay_buffer_size=replay_buffer_size
        self.batch_size=batch_size
        self.sync_target_episode_count=sync_target_episode_count
        self.episode=0
        self.episode_count=episode_count

    def fn_init(self,gameboard):
        self.gameboard=gameboard
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # In this function you could set up and initialize the states, actions, the Q-networks, experience replay buffer and storage for the rewards
        # You can use any framework for constructing the networks, for example pytorch or tensorflow
        # This function should not return a value, store Q network etc as attributes of self

        # Useful variables: 
        # 'gameboard.N_row' number of rows in gameboard
        # 'gameboard.N_col' number of columns in gameboard
        # 'len(gameboard.tiles)' number of different tiles
        # 'self.alpha' the learning rate for stochastic gradient descent
        # 'self.episode_count' the total number of episodes in the training
        # 'self.replay_buffer_size' the number of quadruplets stored in the experience replay buffer

    def fn_load_strategy(self,strategy_file):
        pass
        # TO BE COMPLETED BY STUDENT
        # Here you can load the Q-network (to Q-network of self) from the strategy_file

    def fn_read_state(self):
        pass
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # In this function you could calculate the current state of the gane board
        # You can for example represent the state as a copy of the game board and the identifier of the current tile
        # This function should not return a value, store the state as an attribute of self

        # Useful variables: 
        # 'self.gameboard.N_row' number of rows in gameboard
        # 'self.gameboard.N_col' number of columns in gameboard
        # 'self.gameboard.board[index_row,index_col]' table indicating if row 'index_row' and column 'index_col' is occupied (+1) or free (-1)
        # 'self.gameboard.cur_tile_type' identifier of the current tile that should be placed on the game board (integer between 0 and len(self.gameboard.tiles))

    def fn_select_action(self):
        pass
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # Choose and execute an action, based on the output of the Q-network for the current state, or random if epsilon greedy
        # This function should not return a value, store the action as an attribute of self and exectute the action by moving the tile to the desired position and orientation

        # Useful variables: 
        # 'self.epsilon' parameter epsilon in epsilon-greedy policy
        # 'self.epsilon_scale' parameter for the scale of the episode number where epsilon_N changes from unity to epsilon

        # Useful functions
        # 'self.gameboard.fn_move(tile_x,tile_orientation)' use this function to execute the selected action
        # The input argument 'tile_x' contains the column of the tile (0 < tile_x < self.gameboard.N_col)
        # The input argument 'tile_orientation' contains the number of 90 degree rotations of the tile (0 < tile_orientation < # of non-degenerate rotations)
        # The function returns 1 if the action is not valid and 0 otherwise
        # You can use this function to map out which actions are valid or not

    def fn_reinforce(self,batch):
        pass
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # Update the Q network using a batch of quadruplets (old state, last action, last reward, new state)
        # Calculate the loss function by first, for each old state, use the Q-network to calculate the values Q(s_old,a), i.e. the estimate of the future reward for all actions a
        # Then repeat for the target network to calculate the value \hat Q(s_new,a) of the new state (use \hat Q=0 if the new state is terminal)
        # This function should not return a value, the Q table is stored as an attribute of self

        # Useful variables: 
        # The input argument 'batch' contains a sample of quadruplets used to update the Q-network

    def fn_turn(self):
        if self.gameboard.gameover:
            self.episode+=1
            if self.episode%100==0:
                print('episode '+str(self.episode)+'/'+str(self.episode_count)+' (reward: ',str(np.sum(self.reward_tots[range(self.episode-100,self.episode)])),')')
            if self.episode%1000==0:
                saveEpisodes=[1000,2000,5000,10000,20000,50000,100000,200000,500000,1000000];
                if self.episode in saveEpisodes:
                    pass
                    # TO BE COMPLETED BY STUDENT
                    # Here you can save the rewards and the Q-network to data files
            if self.episode>=self.episode_count:
                raise SystemExit(0)
            else:
                self.gameboard.fn_restart()
        else:
            # Select and execute action (move the tile to the desired column and orientation)
            self.fn_select_action()
            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to copy the old state into the variable 'old_state' which is later stored in the ecperience replay buffer

            # Drop the tile on the game board
            reward=self.gameboard.fn_drop()

            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to add the current reward to the total reward for the current episode, so you can save it to disk later

            # Read the new state
            self.fn_read_state()

            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to store the state in the experience replay buffer

            if len(self.exp_buffer) >= self.replay_buffer_size:
                # TO BE COMPLETED BY STUDENT
                # Here you should write line(s) to create a variable 'batch' containing 'self.batch_size' quadruplets 
                self.fn_reinforce(batch)

                if self.episode_count % self.sync_target_episode_count == 0:
                    pass
                    # TO BE COMPLETED BY STUDENT
                    # Here you should write line(s) to copy the current network to the target network

class THumanAgent:
    def fn_init(self,gameboard):
        self.episode=0
        self.reward_tots=[0]
        self.gameboard=gameboard

    def fn_read_state(self):
        pass

    def fn_turn(self,pygame):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit(0)
            if event.type==pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.reward_tots=[0]
                    self.gameboard.fn_restart()
                if not self.gameboard.gameover:
                    if event.key == pygame.K_UP:
                        self.gameboard.fn_move(self.gameboard.tile_x,(self.gameboard.tile_orientation+1)%len(self.gameboard.tiles[self.gameboard.cur_tile_type]))
                    if event.key == pygame.K_LEFT:
                        self.gameboard.fn_move(self.gameboard.tile_x-1,self.gameboard.tile_orientation)
                    if event.key == pygame.K_RIGHT:
                        self.gameboard.fn_move(self.gameboard.tile_x+1,self.gameboard.tile_orientation)
                    if (event.key == pygame.K_DOWN) or (event.key == pygame.K_SPACE):
                        self.reward_tots[self.episode]+=self.gameboard.fn_drop()