#from gym.utils import seeding
import gymnasium as gym
import math
import numpy as np
import os
#import random
import matplotlib.pyplot as plt
from gym.utils import seeding

from gymnasium.spaces import Box, Discrete

class SimplePursuringDiscreteEn(gym.Env):

    def NewRestart(self):    

        self.fi_start = 2*math.pi*np.random.rand()
        self.xx_start = 2*np.random.rand() - np.float(1)
        self.yy_start = 2*np.random.rand() - np.float(1)
          
        self.fi_Tgt_start = 2*math.pi*np.random.rand()
        self.xTgt_start = np.random.rand()-np.float(0.5)
        self.yTgt_start = np.random.rand()-np.float(0.5)
        
        self.FullReward = float(0)
        
    def __init__( self, isNewRestart=True, UseSeed=10, MaximumSteps=500):

        np.random.seed(UseSeed)        

        self.num_episodes = int(0)
        self.MaximumSteps = int(MaximumSteps)
        self.verbose = int(0)
        self.isNewRestart = isNewRestart
        
        self.V = np.float(0.006)
        self.VTgt = np.float(0.001)
        self.MaxAbsAction = np.float(0.1);
        self.i_D = 0 
        
        self.NewRestart()
        
        self.x_Data = np.zeros( (self.MaximumSteps,1) )
        self.y_Data = np.zeros( (self.MaximumSteps,1) )

        self.xTgt_Data = np.zeros( (self.MaximumSteps,1) )
        self.yTgt_Data = np.zeros( (self.MaximumSteps,1) )

        self.FireDistance2 = np.float(0.05)
        self.FireReward = np.float(50)
        self.StepTax = np.float(0.3)
        self.ShappingRatio = np.float(1.0)
        self.ShappingGamma = np.float(1.0)

        self.observation_space = Box(
            low =np.array([-10.0, -10.0, -10.0, -10.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32), 
            high=np.array([+10.0, +10.0, +10.0, +10.0, +1.0, +1.0, +1.0, +1.0, +1.0, +1.0], dtype=np.float32), 
            dtype=np.float32
        )

        self.action_space = Discrete(3)    
    
        with open("Score.txt","w") as file:
            file.write("0\t0\t0")

    def reset(self, seed=0): 
               
  
        # Reset counter   
        self.episode_reward = 0 
        self.num_iterations = 0
        self.num_episodes += 1
               
        self.Score = 0
         
        if  self.isNewRestart:
            self.NewRestart()
    
        self.fi = self.fi_start
        self.fi_Tgt = self.fi_Tgt_start
        self.xx = self.xx_start
        self.yy = self.yy_start
        self.xTgt = self.xTgt_start
        self.yTgt = self.yTgt_start
    
        self.i_D = int(0);
        self.x_Data[self.i_D] = self.xx
        self.y_Data[self.i_D] = self.yy
        
        
        self.xTgt_Data[self.i_D] = self.xTgt
        self.yTgt_Data[self.i_D] = self.yTgt
        
        self.iD = 0
        
        self.FullReward = float(0)
        
        obs = np.array([self.xx, self.yy,  
                        self.xTgt, self.yTgt,
                        math.cos(self.fi_Tgt), math.sin(self.fi_Tgt),
                        math.cos(self.fi), math.sin(self.fi),
                        math.cos(self.fi_Tgt-self.fi), math.sin(self.fi_Tgt-self.fi)], dtype=np.float32)    
        
        return (obs, dict())
     
        
     
    def step(self, action):
        
        #print('step')
        
        
        
        PreviousDistance2 = ( (self.xx-self.xTgt)**2 + (self.yy-self.yTgt)**2 )

        act = np.float(0)

        #print(action)

        if action == 0 :
           act = self.MaxAbsAction

        if action == 2 :
           act = -self.MaxAbsAction
       
        self.fi += act
        self.xx += self.V * math.cos(self.fi)
        self.yy += self.V * math.sin(self.fi)

        self.i_D += 1

        self.xTgt += self.VTgt * math.cos(self.fi_Tgt)
        self.yTgt += self.VTgt * math.sin(self.fi_Tgt)

        self.x_Data[self.i_D] = self.xx
        self.y_Data[self.i_D] = self.yy

        self.xTgt_Data[self.i_D] = self.xTgt
        self.yTgt_Data[self.i_D] = self.yTgt
    
        NextDistance2 = ( (self.xx-self.xTgt)**2 + (self.yy-self.yTgt)**2 )
    
        reward = self.ShappingRatio * (self.ShappingGamma*PreviousDistance2 - NextDistance2) / (2000 * self.V**2) - self.StepTax 
    
    
        next_obs = np.array([self.xx, self.yy,
                        self.xTgt, self.yTgt, 
                        math.cos(self.fi_Tgt), math.sin(self.fi_Tgt),
                        math.cos(self.fi), math.sin(self.fi),                        
                        math.cos(self.fi_Tgt-self.fi), math.sin(self.fi_Tgt-self.fi)], 
                            dtype=np.float32)       
    
        done = self.i_D >= self.MaximumSteps-1 or NextDistance2 < self.FireDistance2
    
        if NextDistance2 < self.FireDistance2:
            reward += self.FireReward
            
            if self.verbose != 0:
                print('Bah!')
       
        self.FullReward += reward 
       
       
        if done:
           self.Score = self.MaximumSteps-1 - self.i_D
           
           with open("Score.txt","a") as file:
                file.write( "\n" + str(self.num_episodes) + "\t" + str(self.FullReward) + "\t" + str(self.Score) )
       
        if self.verbose != 0 and done: 
            #print(self.verbose)
            
            if self.num_episodes % 1 == 0:
                self.show(self.num_episodes)
        
        if self.verbose != 0 and done: 
            print(self.FullReward)
        
        return next_obs, reward, done, False, dict()

    def show(self, episode):
            plt.plot(self.x_Data[0:self.i_D],self.y_Data[0:self.i_D])
            plt.plot(self.xTgt_Data[0:self.i_D],self.yTgt_Data[0:self.i_D])
            plt.axis([-3,3,-3,3])
            #plt.title('Episode='+ str(episode) + '  Length=' + str(self.i_D+1) + '  FullReward=' + str(self.FullReward) + '  Score=' + str(self.Score) )
            plt.title('Episode='+ str(episode)  + '  FullReward=' + str(self.FullReward) + '  Score=' + str(self.Score) )
           #plt.show()
            
            os.makedirs('frames', exist_ok=True)
            frame_path = f'frames/episode_{episode:07d}.png'
            plt.savefig(frame_path)
            plt.close()
            
    def render(self, mode='human'):
        raise NotImplementedError('Please use custom, external rendering.')
        pass            
            
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        #self.show()
        print('End of training')