# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 17:05:59 2022

@author: Fari
"""

import numpy as np
import math
from DDPG_Torch import Agent
from matplotlib import pyplot as plt 
#import os
#import scipy.io

#------------------------------------------------------------------------------------------------------------------------
#np.random.seed(1375)

class environment:
    def __init__(self,n_state,n_agent,n_RB):
        self.n_state = n_state
        self.n_agent=n_agent
        self.n_RB_Macro = n_RB
        self.n_RB = n_RB
        self.Macro_position = [[500 , 500]] #center of Fig
        self.Micro_position = [[250, 250]]
        #self.Micro_position = [[250, 250],[250, 750],[750, 750],[750, 250]]
        self.n_macro = len(self.Macro_position)
        self.n_micro = len(self.Micro_position)
        self.n_BS = self.n_macro + self.n_micro
        self.h_bs = 25
        self.h_ms = 1.5
        self.shadow_std=8
        self.Decorrelation_distance = 50
        self.time_slow = 0.1
        self.velocity = 36 #km/h
        self.V2I_Shadowing = np.random.normal(0, 8, 1)
        self.delta_distance = self.velocity * self.time_slow
        self.veh_power = np.zeros([n_agent])
        self.sig2_dbm = -84
        self.sig2 = 10 ** (self.sig2_dbm-30/10)
        self.BW = 15 #KHz
        self.min_rate = 300
        self.bsAntGain = 8
        self.bsNoiseFigure = 5
        self.vehAntGain = 3
        self.vehNoiseFigure = 9
        self.WiFi_level_min = 6 # 6KB/ms == 6MB/s
        self.rate_level_min = 100 # 100KB/ms == 100MB/s
        self.V2I_signal = np.zeros([n_agent])
        self.interference= np.zeros([n_agent])  
        self.pos_veh = np.zeros([n_agent,2], dtype=np.float16) #position_vehicle[x,y]
        self.dir_veh = np.zeros([n_agent]) #direction of vehicle
        self.veh_BS_allocate = np.zeros([n_agent], dtype=np.int32)
        self.veh_RB = np.zeros([n_agent,n_RB], dtype=np.int32)
        self.veh_num_BS = np.zeros([n_agent,2], dtype=np.int32)
        self.state = np.zeros([self.n_agent,self.n_state], dtype=np.float16)
        self.new_state = np.zeros([self.n_agent,self.n_state], dtype=np.float16)
        self.duty = np.zeros([n_agent], dtype=np.float16)

    def macro_allocate(self,position_veh):
        n_BS = self.n_macro
        dis_all = np.zeros([n_BS])
        for i_BS in range(n_BS):
            Macro_position = self.Macro_position[i_BS]
            d1 = abs(position_veh[0] - Macro_position[0])     
            d2 = abs(position_veh[1] - Macro_position[1])
            dis_all[i_BS] = math.hypot(d1, d2)
            
        return np.argmin(dis_all)
    
    def micro_allocate(self,position_veh):
        n_BS = self.n_micro
        dis_all = np.zeros([n_BS])
        for i_BS in range(n_BS):
            Micro_position = self.Micro_position[i_BS]
            d1 = abs(position_veh[0] - Micro_position[0])     
            d2 = abs(position_veh[1] - Micro_position[1])
            dis_all[i_BS] = math.hypot(d1, d2)
            
        return np.argmin(dis_all)
        

    def get_path_loss_Macro(self,position_veh,i_macro):
            Macro_position = self.Macro_position[i_macro]
            d1 = abs(position_veh[0] - Macro_position[0]) 
            d2 = abs(position_veh[1] - Macro_position[1]) 
            distance = math.hypot(d1, d2)
            r = math.sqrt((distance ** 2) + ((self.h_bs - self.h_ms) ** 2)) / 1000
            if r < 25 : r = 25
            Loss = 128.1 + 37.6 * np.log10(r)
            return Loss
    
    def get_path_loss_Micro(self,position_veh,i_micro):
            Micro_position = self.Micro_position[i_micro]
            d1 = abs(position_veh[0] - Micro_position[0]) 
            d2 = abs(position_veh[1] - Micro_position[1]) 
            distance = math.hypot(d1, d2)
            distance = math.hypot(d1, d2)
            r = math.sqrt((distance ** 2) + ((self.h_bs - self.h_ms) ** 2)) / 1000
            if r < 25 : r = 25
            Loss = 128.1 + 37.6 * np.log10(r)
            return Loss
        
    def get_shadowing(self,delta_distance, shadowing):
            self.R = np.sqrt(0.5 * np.ones([1, 1]) + 0.5 * np.identity(1))
            return np.multiply(np.exp(-1 * (delta_distance / self.Decorrelation_distance)), shadowing) \
                   + np.sqrt(1 - np.exp(-2 * (delta_distance / self.Decorrelation_distance))) * np.random.normal(0, 8, 1) #فرمول shadowings
                   

    def get_reward(self,WiFi_level,rate_level,veh_RB,veh_RB_BS,veh_micro,i_step):
        reward = 0
        for i_BS in range(self.n_BS):
            for i_RB in range(self.n_RB):
                if np.sum(veh_RB_BS[:,i_RB,i_BS]) > 1 : 
                   reward += -10
        if WiFi_level > self.WiFi_level_min : reward += (rate_level/self.rate_level_min)
        return reward
        
    def get_state(self,veh_RB_power,veh_BS,veh_RB,WiFi_rate,n_duty,i_agent,i_step):
        if veh_BS[i_agent,i_step] == 1 : 
            i_macro = environment.macro_allocate(self,self.pos_veh[i_agent]) 
            i_micro = 0
            self.state[i_agent,0] = environment.get_path_loss_Macro(self,self.pos_veh[i_agent],i_macro)
            self.veh_num_BS[i_agent,0] = -1
            self.veh_num_BS[i_agent,1] = i_macro
           
        else : 
            i_micro = environment.micro_allocate(self,self.pos_veh[i_agent])
            i_macro = 0
            self.state[i_agent,0] = environment.get_path_loss_Micro(self,self.pos_veh[i_agent],i_micro)
            self.veh_num_BS[i_agent,0] = i_micro
            self.veh_num_BS[i_agent,1] = -1
    
        
        self.state[i_agent,1] = environment.compute_rate(self,veh_RB_power,veh_BS,veh_RB,WiFi_level,i_agent,i_step)
        self.state[i_agent,2] = environment.get_interference(self,veh_RB_power,veh_BS,veh_RB,i_macro,i_micro,i_agent,i_step) * (10**16)
        self.state[i_agent,3] = WiFi_rate[i_agent,i_step]
        
        return self.state[i_agent], self.veh_num_BS[i_agent]
    
    def compute_rate(self,veh_RB_power,veh_BS,veh_RB,WiFi_level,i_agent,i_step): 
        self.V2I_signal[i_agent] = 0
        if veh_BS[i_agent,i_step] == 1 :
           i_macro = environment.macro_allocate(self,self.pos_veh[i_agent])
           i_micro = 0
           veh_gain = environment.get_path_loss_Macro(self,self.pos_veh[i_agent],i_macro)
           for i_RB in range(self.n_RB) :
               self.V2I_signal[i_agent] += (veh_RB[i_agent,i_RB,i_step]) * (10**((veh_RB_power[i_agent,i_RB,i_step]-veh_gain + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure)/10))
           veh_rate = self.BW * np.log2(1 + np.divide(self.V2I_signal[i_agent], (environment.get_interference(self,veh_RB_power,veh_BS,veh_RB,i_macro,i_micro,i_agent,i_step) + self.sig2))) 
        
        elif veh_BS[i_agent,i_step] == 0 :
           i_micro = environment.micro_allocate(self,self.pos_veh[i_agent])
           i_macro = 0
           veh_gain = environment.get_path_loss_Micro(self,self.pos_veh[i_agent],i_micro)
           for i_RB in range(self.n_RB):
               self.V2I_signal[i_agent] += (veh_RB[i_agent,i_RB,i_step]) * (10**((veh_RB_power[i_agent,i_RB,i_step]-veh_gain + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure)/10))
           veh_rate = self.BW * np.log2(1 + np.divide(self.V2I_signal[i_agent], (environment.get_interference(self,veh_RB_power,veh_BS,veh_RB,i_macro,i_micro,i_agent,i_step) + self.sig2)))      
        return veh_rate
    
    def get_interference(self,veh_RB_power,veh_BS,veh_RB,i_macro,i_micro,i_agent,i_step):
        self.interference[i_agent] = 0
        if veh_BS[i_agent,i_step] == 1 :
           for i_agent_plus in range(n_agent): 
               if i_agent_plus == i_agent : continue
               if veh_BS[i_agent_plus,i_step-1] == 1 :
                   for i_RB in range(self.n_RB) :
                       self.interference[i_agent] += veh_RB[i_agent_plus,i_RB,i_step-1] * veh_RB[i_agent,i_RB,i_step-1] * (10**((veh_RB_power[i_agent_plus,i_RB,i_step-1] - environment.get_path_loss_Macro(self,self.pos_veh[i_agent_plus],i_macro) + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure)/10))
        if veh_BS[i_agent,i_step] == 0 :
           for i_agent_plus in range(n_agent): 
               if i_agent_plus == i_agent : continue
               if veh_BS[i_agent_plus,i_step-1] == 0 :
                  for i_RB in range(self.n_RB) :
                      self.interference[i_agent] += veh_RB[i_agent_plus,i_RB,i_step-1] * veh_RB[i_agent,i_RB,i_step-1] * (10**((veh_RB_power[i_agent_plus,i_RB,i_step-1] - environment.get_path_loss_Micro(self,self.pos_veh[i_agent_plus],i_micro) + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure)/10))
        return self.interference[i_agent]
                
        

    def make_start(self,i_agent):
        self.AoI_veh = np.ones([self.n_agent], dtype=np.int64)*100
        self.AoI_WiFi = np.ones([self.n_agent], dtype=np.int64)*100
        self.pos_veh[i_agent,0] = np.round(np.random.uniform(0,1000), 2)
        self.pos_veh[i_agent,1] = np.round(np.random.uniform(0,1000), 2)
        self.dir_veh[i_agent] = np.random.choice ((1,2,3,4),1) #It means: (1='up',2='right',3='down',4='left')
        self.veh_power[i_agent] = np.random.uniform(1,30)
        veh_BS = np.random.randint(0,2)
        
        if veh_BS == 1 : 
            i_Macro = environment.macro_allocate(self,self.pos_veh[i_agent])
            self.state[i_agent,0] = environment.get_path_loss_Macro(self,self.pos_veh[i_agent],i_Macro)
            self.state[i_agent,1] = 0
            self.state[i_agent,2] = 0
            self.state[i_agent,3] = np.round(np.random.uniform(6,12))
            self.veh_num_BS[i_agent,0] = -1 #It means vehicle no connected to Micro
            self.veh_num_BS[i_agent,1] = i_Macro
            self.duty[i_agent] = 1

            
        else : 
            i_Micro = environment.micro_allocate(self,self.pos_veh[i_agent])
            self.state[i_agent,0] = environment.get_path_loss_Micro(self,self.pos_veh[i_agent],i_Micro)
            self.state[i_agent,1] = 0
            self.state[i_agent,2] = 0
            self.state[i_agent,3] = np.round(np.random.uniform(6,12))
            self.veh_num_BS[i_agent,0] = i_Micro
            self.veh_num_BS[i_agent,1] = -1 #It means vehicle no connected to Macro
            self.duty[i_agent] = np.random.uniform(0,1)

            
        
        return self.state[i_agent], self.veh_num_BS[i_agent], self.duty[i_agent]
    
    
    def mobility_veh(self):
        for i_agent in range(n_agent):
            if (self.dir_veh[i_agent]) == 1 :
                self.pos_veh[i_agent,0] = self.pos_veh[i_agent,0] 
                self.pos_veh[i_agent,1] = self.pos_veh[i_agent,1] + 0.1
            if (self.dir_veh[i_agent]) == 2 :
                self.pos_veh[i_agent,0] = self.pos_veh[i_agent,0] 
                self.pos_veh[i_agent,1] = self.pos_veh[i_agent,1] - 0.1
            if (self.dir_veh[i_agent]) == 3 :
                self.pos_veh[i_agent,0] = self.pos_veh[i_agent,0] + 0.1
                self.pos_veh[i_agent,1] = self.pos_veh[i_agent,1] 
            if (self.dir_veh[i_agent]) == 4 :
                self.pos_veh[i_agent,0] = self.pos_veh[i_agent,0] - 0.1
                self.pos_veh[i_agent,1] = self.pos_veh[i_agent,1] 
                
                
    def Age_of_information(self,rate_level,WiFi_rate):
        if rate_level > self.rate_level_min :
            self.AoI_veh[i_agent] = 1
        else :
            self.AoI_veh[i_agent] +=1
            if self.AoI_veh[i_agent] >= 100 : 
                self.AoI_veh[i_agent] = 100
        
        if WiFi_rate > self.WiFi_level_min :
            self.AoI_WiFi[i_agent] = 1
        else :
            self.AoI_WiFi[i_agent] +=1
            if self.AoI_WiFi[i_agent] >= 100 : 
                self.AoI_WiFi[i_agent] = 100
        
        return  self.AoI_veh[i_agent], self.AoI_WiFi[i_agent]
    
    def G_functionself(self,G_input):
        if G_input >= 0 :
            G_output = 1
        else : G_output = 0
        return G_output
    
    def RB_BS_allocate(self,veh_RB,veh_RB_BS,veh_BS,i_step):
        for i_agent in range(self.n_agent):
            if veh_BS[i_agent,i_step] == 0 :
                i_micro = environment.micro_allocate(self,self.pos_veh[i_agent])
                for i_RB in range(self.n_RB):
                    veh_RB_BS[i_agent,i_RB,i_micro+self.n_macro] = veh_RB[i_agent,i_RB,i_step]
            if veh_BS[i_agent,i_step] == 1 :
                i_macro = environment.macro_allocate(self,self.pos_veh[i_agent])
                for i_RB in range(self.n_RB):
                    veh_RB_BS[i_agent,i_RB,i_macro] = veh_RB[i_agent,i_RB,i_step] 
        return veh_RB_BS
    
    def check_constrain(self,veh_RB,veh_RB_BS,i_step):
        for i_BS in range(self.n_BS):
            for i_RB in range(self.n_RB):
                if np.sum(veh_RB_BS[:,i_RB,i_BS]) > 1 : 
                   for i_agent in range(self.n_agent):
                       if np.sum(veh_RB_BS[i_agent,:,i_BS]) > 1 :
                           veh_RB_BS[i_agent,i_RB,i_BS] = 0
                           veh_RB[i_agent,i_RB,i_step] = 0
                if np.sum(veh_RB_BS[:,i_RB,i_BS]) > 1 :
                    veh_RB_BS[:,i_RB,i_BS] = 0
                    veh_RB[:,i_RB,i_step] = 0
        return veh_RB
        
        
                
                

#--------------------------------------------------------------------------------------------------------------------------       
n_episode = 1
n_step = 1
n_agent = 5

n_Macro = 1
n_Micro = 1

n_RB = 12        
n_state = 4
n_action = 1 + (2*n_RB) + 1
max_power_Macro = 30 # Vehicle maximum power is 1 watt 
max_power_Micro = 30
n_mode = 2 # Macro/Micro mode
n_BS = n_Macro + n_Micro
n_RB_Macro = n_RB
n_RB_Micro = n_RB
size_packet = 1000






agent = Agent(alpha=0.0001,beta=0.001,input_dims=n_state,tau=0.005,n_actions=n_action,gamma=0.99,max_size=1000000,C_fc1_dims=1024,C_fc2_dims=512,C_fc3_dims=256,A_fc1_dims=1024,A_fc2_dims=512,batch_size=64,n_agents=n_agent)
env = environment(n_state=n_state,n_agent=n_agent,n_RB=n_RB)

i_episode_matrix = np.zeros ([n_episode], dtype=np.int16)
reward_per_episode = np.zeros ([n_episode], dtype=np.float16)
reward_mean_all_episode = np.zeros([n_episode], dtype=np.float16)
rate_mean_all_episode = np.zeros([n_episode], dtype=np.float16)
rate_level_mean_all = np.zeros([n_episode], dtype=np.float16)
WiFi_level_mean_all = np.zeros([n_episode], dtype=np.float16)
rate_per_episode = np.zeros ([n_agent,n_episode], dtype=np.float16)
rate_level_per_episode = np.zeros ([n_agent,n_episode], dtype=np.float16)
WiFi_level_per_episode = np.zeros ([n_agent,n_episode], dtype=np.float16)

#reward_mem = []
#lets go(Start episode)---------------------------------------------------------------------------------------------------
for i_episode in range(n_episode):  
    done = False
    packet_done = np.zeros([n_agent,n_step], dtype=np.int32)
    i_episode_matrix[i_episode] = i_episode
    #initialize parameters------------------------------------------------------------------------------
    state = np.zeros([n_agent,n_state], dtype=np.float16)
    new_state = np.zeros([n_agent,n_state], dtype=np.float16)
    RB_Micro = np.zeros([n_Micro,n_RB_Micro], dtype=np.int16)
    RB_Macro = np.zeros([n_RB_Macro], dtype=np.int16)
    veh_Micro = np.zeros([n_agent,n_step], dtype=np.int32)
    veh_Macro = np.zeros([n_agent,n_step], dtype=np.int32)
    veh_RB_power = np.zeros([n_agent,n_RB,n_step])
    veh_RB = np.zeros([n_agent,n_RB,n_step],dtype=np.int16)
    veh_num_BS = np.zeros([n_agent,2], dtype=np.int32)
    duty = np.zeros([n_agent], dtype=np.float16)  
    n_duty =  np.zeros([n_agent,n_step], dtype=np.int32)  
    i_step_matrix = np.zeros ([n_step], dtype=np.int16)
    reward_per_step = np.zeros ([n_step], dtype=np.float16)
    rate_per_step = np.zeros ([n_agent,n_step], dtype=np.float16)
    AoI_veh = np.ones([n_agent], dtype=np.int64)*100
    AoI_WiFi = np.ones([n_agent], dtype=np.int64)*100
    veh_BS_allocate = np.zeros([n_agent], dtype=np.int32)
    veh_gain = np.zeros([n_agent,n_step], dtype = np.float16)
    veh_BS = np.zeros([n_agent,n_step], dtype=np.int32)
    veh_flag = np.zeros([n_agent], dtype=np.int32)
    veh_data = np.zeros([n_agent,n_step], dtype = np.float16)
    veh_data[:,0] = size_packet
    WiFi_level = np.zeros([n_agent,n_step])
    WiFi_rate = np.zeros([n_agent,n_step])
    rate_level = np.zeros([n_agent,n_step])
    #make start Environment------------------------------------------------------------------------------
    for i_micro in range(n_Micro) :
        RB_Micro[i_micro] = np.random.choice(n_RB_Micro, n_RB_Micro, replace=False)
    RB_Macro = np.random.choice(n_RB_Macro, n_RB_Macro, replace=False)
    for i_agent in range(n_agent):
        state[i_agent], veh_num_BS[i_agent], duty[i_agent] = env.make_start(i_agent)
    #Start step-----------------------------------------------------------------------------------------
    for i_step in range(n_step):
        action = np.zeros([n_agent,n_action], dtype=np.float16)
        reward = np.zeros ([0], dtype=np.float16)
        veh_RB_BS = np.zeros([n_agent,n_RB,n_BS],dtype=np.int16)
        i_step_matrix[i_step] = i_step
        env.mobility_veh()
        i_RB_Macro = 0
        i_RB_Micro = np.zeros([n_Micro], dtype=np.int32)
        
        for i_agent in range(n_agent):
            packet_done[i_agent,i_step] = 0
            veh_gain[i_agent,i_step] = state[i_agent,0]
            WiFi_rate[i_agent,i_step] = np.random.uniform(6,12)
            if i_step !=0 :
                veh_data[i_agent,i_step] = veh_data[i_agent,i_step-1] - (n_duty[i_agent,i_step] * (rate_per_step[i_agent,i_step-1]))
                if veh_data[i_agent,i_step] <= 0 :
                    veh_data[i_agent,i_step] = size_packet
                    veh_flag[i_agent] += 1
                    packet_done[i_agent,i_step] = 1
        #state process and reshape------------------------------------------------------------
        state_shape = np.reshape(state,(1,n_state*n_agent))
        if np.round(np.ndarray.max(state_shape)) == 0 : state_shape = np.zeros([1,n_state*n_agent])
        else : state_shape = state_shape / np.ndarray.max(state_shape)
        #action process-----------------------------------------------------------------
        action_choose = agent.choose_action(state_shape)
        action_choose = np.clip(action_choose, 0.000, 0.999)
        for i_agent in range(n_agent):
            #Allocation-------------------------------------------------------------------
            if veh_num_BS[i_agent,1] == -1 : veh_Micro[i_agent,i_step] = veh_num_BS[i_agent,0]
            if veh_num_BS[i_agent,0] == -1 : veh_Macro[i_agent,i_step] = veh_num_BS[i_agent,1]
            #BS & RB & duty-cycle allocation--------------------------------------------------------
            action[i_agent,0] = int((action_choose[0,0+i_agent*n_action]) * n_mode) # chosen type of BS
            if action[i_agent,0] == 0 : #Allocate to Micro
                veh_BS[i_agent,i_step] = 0
                for i in range(1,n_RB+1): #Allocation RB
                    action[i_agent,i] = int((action_choose[0,i+i_agent*n_action]) * 2) 
                    veh_RB[i_agent,i-1,i_step] = action[i_agent,i]
                for i in range(n_RB+1,n_RB+n_RB+1): #Allocation Power 
                    action[i_agent,i] = np.round(np.clip(action_choose[0,i+(i_agent*n_action)] * max_power_Micro, 1, max_power_Micro))  # power selected by veh
                    veh_RB_power[i_agent,i-(n_RB+1),i_step] =  action[i_agent,i]
                    
                action[i_agent,n_RB+n_RB+1] = action_choose[0,n_RB+n_RB+1+(i_agent*n_action)] #Duty-cycle
                duty[i_agent] = action[i_agent,n_RB+n_RB+1]
                WiFi_level[i_agent,i_step] = (1 - duty[i_agent]) * (WiFi_rate[i_agent,i_step])
                i_RB_Micro[veh_Micro[i_agent]] += 1
                
            elif action[i_agent,0] == 1 : #Allocate to Macro
                veh_BS[i_agent,i_step] = 1
                for i in range(1,n_RB+1): #Allocation RB
                    action[i_agent,i] = int((action_choose[0,i+i_agent*n_action]) * 2) 
                    veh_RB[i_agent,i-1,i_step] = action[i_agent,i]
                for i in range(n_RB+1,n_RB+n_RB+1): #Allocation Power 
                    action[i_agent,i] = np.round(np.clip(action_choose[0,i+i_agent*n_action] * max_power_Macro, 1, max_power_Macro))  # power selected by veh
                    veh_RB_power[i_agent,i-(n_RB+1),i_step] =  action[i_agent,i]
                    
                action[i_agent,n_RB+n_RB+1] = 1 #Duty-cycle
                duty[i_agent] = action[i_agent,n_RB+n_RB+1]
                WiFi_level[i_agent,i_step] = WiFi_rate[i_agent,i_step]
                i_RB_Macro +=1
                
        #Check Constrain---------------------------------------------------------------------------------------------
        veh_RB_BS = env.RB_BS_allocate(veh_RB,veh_RB_BS,veh_BS,i_step)
        #veh_RB = env.check_constrain(veh_RB,veh_RB_BS,i_step)
        
        #Calculate parameters---------------------------------------------------------------------------------------------
        for i_agent in range(n_agent):
            rate_per_step[i_agent,i_step] = env.compute_rate(veh_RB_power,veh_BS,veh_RB,WiFi_level,i_agent,i_step)
            rate_level[i_agent,i_step] = duty[i_agent] * (rate_per_step[i_agent,i_step])
            AoI_veh[i_agent], AoI_WiFi[i_agent] = env.Age_of_information(WiFi_level[i_agent,i_step],rate_level[i_agent,i_step])
            n_duty[i_agent,i_step] = np.round(duty[i_agent] * n_step)
        #new state process and reshape-------------------------------------------------------------------------
        for i_agent in range (n_agent):
            new_state[i_agent], veh_num_BS[i_agent] = env.get_state(veh_RB_power,veh_BS,veh_RB,WiFi_rate,n_duty,i_agent,i_step)
        new_state_shape = np.reshape(new_state,(1,n_state*n_agent))
        if np.round(np.ndarray.max(new_state_shape)) == 0 : new_state_shape = np.zeros([1,n_state*n_agent])
        else : new_state_shape = new_state_shape / np.ndarray.max(new_state_shape)
        #Calculate reward and store memory and learn
        reward = env.get_reward(np.mean(WiFi_level[:,i_step]),np.mean(rate_level[:,i_step]),veh_RB,veh_RB_BS,veh_Micro,i_step)
        if i_step == n_step - 1 : done = True
        agent.remember(state_shape, action_choose, reward, new_state_shape, done)
        reward_per_step[i_step] = reward
        print('episode:',i_episode,' step:',i_step,' reward:',reward)
        state = new_state.copy()  
        agent.learn()
#plot process-------------------------------------------------------------------------------------        
    reward_per_episode[i_episode] = np.mean(reward_per_step[:])     
    
    for i_agent in range(n_agent):
        WiFi_level_per_episode[i_agent,i_episode] = np.mean(WiFi_level[i_agent,:])
        rate_level_per_episode[i_agent,i_episode] = np.mean(rate_level[i_agent,:])
for i_episode in range(n_episode) :
    rate_level_mean_all[i_episode] = np.mean(rate_level_per_episode[:,i_episode])
    WiFi_level_mean_all[i_episode] = np.mean(WiFi_level_per_episode[:,i_episode])
    
plt.plot(i_episode_matrix,reward_per_episode) 
plt.show()
plt.plot(i_episode_matrix,rate_level_mean_all) 
plt.show()
plt.plot(i_episode_matrix,WiFi_level_mean_all) 



      
        
        
        