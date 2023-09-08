# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 16:55:39 2022

@author: Fari
"""

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DeepQNetwork(nn.Module):
    def __init__(self, epsilon, epsilon_min, epsilon_dec, gamma, lr, mem_size, batch_size, input_dims, n_dims, fc1_dims,fc2_dims,fc3_dims,output_dims):
        super(DeepQNetwork, self).__init__()
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec
        self.gamma = gamma
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.output_dims = output_dims
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims,self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims,self.fc3_dims)
        self.fc4 = nn.Linear(self.fc3_dims,self.output_dims)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.n_state = n_dims
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.state_memory = np.zeros([self.mem_size,self.n_state], dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size,dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size,dtype=np.float32)
        self.new_state_memory = np.zeros([self.mem_size,self.n_state],dtype=np.float32)
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        
    def forward(self,state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = self.fc4(x)
        return x
    
    def choose_action(self,state):
        if np.random.random() > self.epsilon :
            state = np.ndarray.tolist(state)
            state = T.tensor([state]).to(self.device)
            q = DeepQNetwork.forward(self,state)
            action = T.argmax(q).item()
        else :
            action = np.random.randint(self.output_dims)
            
        return action
    
    def store_transition(self,state, action, reward, new_state):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = new_state
        self.mem_cntr +=1
         
    def learn(self):
        if self.mem_cntr < self.batch_size:
           return
       
        max_mem = min(self.mem_cntr, self.mem_size) 
        
        batch = np.random.choice(max_mem, self.batch_size, replace=False) 
        
        batch_index = np.arange(self.batch_size, dtype=np.int32) 
        
        state_batch = T.tensor(self.state_memory[batch]).to(self.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.device)
        #terminal_batch = T.tensor(self.terminal_memory[batch]).to(DQN.device)
        action_batch = self.action_memory[batch]
        
        q_eval = DeepQNetwork.forward(self,state_batch)[batch_index, action_batch]
        q_next = DeepQNetwork.forward(self,new_state_batch)
        #q_next[terminal_batch] = 0.0
        
        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]
        
        loss = self.loss(q_target, q_eval).to(self.device)
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min :
           self.epsilon = self.epsilon - self.epsilon_dec
        else :
           self.epsilon = self.epsilon_min

        
        


