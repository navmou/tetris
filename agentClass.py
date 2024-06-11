import numpy as np
import random
import math
import h5py
import hdfdict
import tensorflow as tf
from collections import deque
import time

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
        self.Q = {}


    def fn_load_strategy(self,strategy_file):
        self.Q = dict(hdfdict.load(f'{strategy_file}.h5'))
        self.Q = self.change_to_int()
        print(f'Number of states in Q matrix: {len(self.Q.keys())}')
    
    def change_to_int(self):
        transformed = {}
        for key in self.Q.keys():
            new_key = key.split(sep="_")
            int_key = tuple([int(float(i)) for i in new_key])
            transformed[int_key] = self.Q[key]
        return transformed
    

    def fn_read_state(self):
        dummy_state = [self.gameboard.cur_tile_type , 0 , 0 , 0 , 0]
        for column in range(self.gameboard.N_col):
            for row in range(self.gameboard.N_row):
                if self.gameboard.board[self.gameboard.N_row-1 - row][column] == 1:
                    dummy_state[column+1] = self.gameboard.N_row - row
                    break
        self.state = tuple(dummy_state)
        
        if self.state not in self.Q.keys():
            if self.state[0] == 0:
                self.Q[self.state] = [0]*7
            elif self.state[0] == 1:
                self.Q[self.state] = [0]*6
            elif self.state[0] == 2:
                self.Q[self.state] = [0]*12
            elif self.state[0] == 3:
                self.Q[self.state] = [0]*3
        
        #print(self.state)
                    
                    
    def fn_execute_action(self):
        if self.state[0] == 0:
            if self.action == 0:
                self.gameboard.fn_move(2 , self.gameboard.tile_orientation)
            elif self.action == 1:
                self.gameboard.fn_move(3 , self.gameboard.tile_orientation)
            elif self.action == 2:
                self.gameboard.fn_move(1 , self.gameboard.tile_orientation)
            elif self.action == 3:
                self.gameboard.fn_move(0 , self.gameboard.tile_orientation)
            elif self.action == 4:
                self.gameboard.fn_move(2 , self.gameboard.tile_orientation+1)
            elif self.action == 5:
                self.gameboard.fn_move(1 , self.gameboard.tile_orientation+1)
            elif self.action == 6:
                self.gameboard.fn_move(0 , self.gameboard.tile_orientation+1)
            
        elif self.state[0] == 1:
            if self.action == 0:
                self.gameboard.fn_move(2 , self.gameboard.tile_orientation)
            elif self.action == 1:
                self.gameboard.fn_move(1 , self.gameboard.tile_orientation)
            elif self.action == 2:
                self.gameboard.fn_move(0 , self.gameboard.tile_orientation)
            elif self.action == 3:
                self.gameboard.fn_move(2 , self.gameboard.tile_orientation+1)
            elif self.action == 4:
                self.gameboard.fn_move(1 , self.gameboard.tile_orientation+1)
            elif self.action == 5:
                self.gameboard.fn_move(0 , self.gameboard.tile_orientation+1)
        
        elif self.state[0] == 2:
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
            
        elif self.state[0] == 3:
            if self.action == 0:
                self.gameboard.fn_move(self.gameboard.tile_x , self.gameboard.tile_orientation)
            elif self.action == 1:
                self.gameboard.fn_move(self.gameboard.tile_x-1 , self.gameboard.tile_orientation)
            elif self.action == 2:
                self.gameboard.fn_move(self.gameboard.tile_x-2 , self.gameboard.tile_orientation)        


    def fn_select_action(self):
        if self.epsilon < np.random.uniform():
            self.action = np.argmax(self.Q[self.state])
            #print(self.Q[self.state])
        else:
            if self.state[0] == 0:
                self.action = np.random.choice(range(0,7))
            elif self.state[0] == 1:
                self.action = np.random.choice(range(0,6))
            elif self.state[0] == 2:
                self.action = np.random.choice(range(0,12))
            elif self.state[0] == 3:
                self.action = np.random.choice(range(0,3))

        self.fn_execute_action()

    
    def fn_reinforce(self,old_state,reward):
    	if self.gameboard.gameover:
    	    self.Q[old_state][self.action] += self.alpha*(self.game_reward_sum - self.Q[old_state][self.action])
    	else:
            self.Q[old_state][self.action] += self.alpha*(reward + 0.9*np.max(self.Q[self.state]) - self.Q[old_state][self.action])

    def fn_turn(self ):
        if self.gameboard.gameover:
            self.episode+=1
            self.game_reward_sum_list.append(self.game_reward_sum)
            self.reward_tots.append(self.game_reward_sum)
            if self.game_reward_sum > 100:
            	hdfdict.dump(self.Q, f"Q_episode{self.episode}.h5")
            self.game_reward_sum = 0
            if self.episode%100==0:
                begin = self.episode-100
                end = self.episode
                print(f'episode {(self.episode)}/{(self.episode_count)} (reward: {np.sum(self.reward_tots[begin:end])})')
                self.moving_average.append(np.sum(self.reward_tots[begin:end])/100)
                print(f'episode {self.episode} reward: {self.game_reward_sum_list[self.episode-1]}')
            
                    
            if self.episode>=self.episode_count:
                hdfdict.dump(self.Q, "Q.h5")
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
            self.fn_select_action()
            old_state = self.state
            reward=self.gameboard.fn_drop()
            self.game_reward_sum += reward
            self.fn_read_state()

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
        self.game_reward_sum = 0
        self.game_reward_sum_list = []
        self.reward_tots = []
        self.moving_average = []

        self.discount = 0.9
        self.target_update_counter = 0
        self.tot_actions = 13


    def fn_init(self,gameboard):
        self.gameboard=gameboard
        self.tile_0_valid_actions = [0,1,2,3,4,5,6]
        self.tile_1_valid_actions = [0,1,2,4,5,6]
        self.tile_2_valid_actions = [0,1,2,4,5,6,7,8,9,10,11,12]
        self.tile_3_valid_actions = [0,1,2]
            
        self.Q = self.create_model()
        
        self.target_Q = self.create_model()
        self.target_Q.set_weights(self.Q.get_weights())
        
        self.replay_memory = deque(maxlen=self.replay_buffer_size)

        
    def create_model(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.Input(shape=(20,)))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(self.tot_actions, activation='linear'))
        model.compile(loss='mse' , optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      metrics=['accuracy'])
        return model
    
        
    def fn_load_strategy(self,strategy_file):
        self.Q = tf.keras.models.load_model(strategy_file)


    def fn_read_state(self):
        self.state = []
        for row in range(self.gameboard.N_row):
            for col in range(self.gameboard.N_col):
                self.state.append(self.gameboard.board[row][col])
        
        if self.gameboard.cur_tile_type == 0:
            self.state += [1,0,0,0]
        elif self.gameboard.cur_tile_type == 1:
            self.state += [0,1,0,0]
        elif self.gameboard.cur_tile_type == 2:
            self.state += [0,0,1,0]
        else:
            self.state += [0,0,0,1]
        
        self.state = np.array(self.state)
        

    def validate_actions(self, actions):
        if self.state[-1] == 1:
            valid_actions = list(actions[0][0:3])+[-10000000]*10
        elif self.state[-2] == 1:
            valid_actions = list(actions[0][0:3])+[-10000000]+list(actions[0][4:])
        elif self.state[-3] == 1:
            valid_actions = list(actions[0][0:3])+[-10000000]+list(actions[0][4:7])+[-10000000]*6
        else:
            valid_actions = list(actions[0][0:7])+[-10000000]*6
        
        return valid_actions

    
    def choose_random_action(self):
        if self.state[-1] == 1:
            random_action = np.random.choice(self.tile_3_valid_actions)
        elif self.state[-2] == 1:
            random_action = np.random.choice(self.tile_2_valid_actions)
        elif self.state[-3] == 1:
            random_action = np.random.choice(self.tile_1_valid_actions)
        elif self.state[-4] == 1:
            random_action = np.random.choice(self.tile_0_valid_actions)
        
        return random_action
        

    def fn_select_action(self):
        if np.random.uniform() > self.epsilon:
            actions = self.Q.predict(np.array(self.state).reshape(-1, *self.state.shape))
            valid_actions = self.validate_actions(actions)
            self.action = np.argmax(valid_actions)
        else:
            self.action = self.choose_random_action()
        
        self.execute_action()
            
        
    def execute_action(self):
        if self.action == 0:
            self.gameboard.fn_move(0 , self.gameboard.tile_orientation)
        elif self.action == 1:
            self.gameboard.fn_move(1 , self.gameboard.tile_orientation)
        elif self.action == 2:
            self.gameboard.fn_move(2 , self.gameboard.tile_orientation)
        elif self.action == 3:
            self.gameboard.fn_move(3 , self.gameboard.tile_orientation)
        elif self.action == 4:
            self.gameboard.fn_move(0 , self.gameboard.tile_orientation+1)
        elif self.action == 5:
            self.gameboard.fn_move(1 , self.gameboard.tile_orientation+1)
        elif self.action == 6:
            self.gameboard.fn_move(2 , self.gameboard.tile_orientation+1)
        elif self.action == 7:
            self.gameboard.fn_move(0 , self.gameboard.tile_orientation+2)
        elif self.action == 8:
            self.gameboard.fn_move(1 , self.gameboard.tile_orientation+2)
        elif self.action == 9:
            self.gameboard.fn_move(2 , self.gameboard.tile_orientation+2)
        elif self.action == 10:
            self.gameboard.fn_move(0 , self.gameboard.tile_orientation+3)
        elif self.action == 11:
            self.gameboard.fn_move(1 , self.gameboard.tile_orientation+3)
        elif self.action == 12:
            self.gameboard.fn_move(2 , self.gameboard.tile_orientation+3)

        

    def fn_reinforce(self,batch):
        current_states = np.array([transition[0] for transition in batch])
        current_qs_list = self.Q.predict(current_states)
        new_current_states = np.array([transition[3] for transition in batch])
        future_qs_list = self.target_Q.predict(new_current_states)
        
        X = []
        y = []
        
        for index, (current_state , action , reward , new_state) in enumerate(batch):
            max_future_q = np.max(future_qs_list[index])
            new_q = reward + self.discount*max_future_q
                  
            current_qs = current_qs_list[index]
            current_qs[action] = new_q
            
            X.append(current_state)
            y.append(current_qs)
            
        self.Q.fit(np.array(X) , np.array(y) , batch_size = self.batch_size , verbose = 0 , shuffle = False)
        
        
        

    def fn_turn(self):
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
                saveEpisodes=[1000,2000,5000,10000,20000,50000,100000,200000,500000,1000000];
                if self.episode in saveEpisodes:
                    pass
                    # TO BE COMPLETED BY STUDENT
                    # Here you can save the rewards and the Q-network to data files
            if self.episode>=self.episode_count:
                self.target_Q.save('model.h5')
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
            current_state = self.state
            self.fn_select_action()
            
            reward=self.gameboard.fn_drop()
            self.game_reward_sum += reward
            self.fn_read_state()

            self.replay_memory.append((current_state , self.action , reward , self.state))

            if len(self.replay_memory) >= self.replay_buffer_size:
                np.random.shuffle(self.replay_memory)
                batch = random.sample(self.replay_memory, self.batch_size)
                self.fn_reinforce(batch)

                if self.episode_count % self.sync_target_episode_count == 0:
                    self.target_Q.set_weights(self.Q.get_weights())


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
