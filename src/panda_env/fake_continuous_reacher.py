import numpy as np
import rospy
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
import sys
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import trajectory_msgs.msg
import gym
from gym.utils import seeding
import os

from gym import utils
import copy
import rospy
from gym import spaces
#from openai_ros.robot_envs import panda_env
from gym.envs.registration import register
import numpy as np
from sensor_msgs.msg import JointState
import os
import rosparam
import rospkg


class FakeReacher(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self, n_actions=3, n_observations=6, position_delta = 0.02, tol = 0.025, render=False):
        
        #super(ReacherEnv, self).__init__()
        
        ros_ws_abspath = rospy.get_param("/panda/ros_ws_abspath", '/home/padmaja/ros/')
        assert ros_ws_abspath is not None, "You forgot to set ros_ws_abspath in your yaml file of your main RL script. Set ros_ws_abspath: \'YOUR/SIM_WS/PATH\'"
        assert os.path.exists(ros_ws_abspath), "The Simulation ROS Workspace path " + ros_ws_abspath + \
                                               " DOESNT exist, execute: mkdir -p " + ros_ws_abspath + \
                                               "/src;cd " + ros_ws_abspath + ";catkin_make"
        
        self.n_actions = n_actions # Number of force values the agent can choose
        low_act = np.array([-1.0]*self.n_actions)
        high_act = np.array([1.0]*self.n_actions)
        self.action_space = spaces.Box(low=low_act, high=high_act)
        
        self.n_observations = n_observations # Number of possible observations
        
        low = np.array([-1.]*self.n_observations)
        high = np.array([1.]*self.n_observations)
        self.observation_space = spaces.Box(low=low, high=high)
        self.rendering = render
        self.position_delta = position_delta
        self.tolerence = tol
        self.init_array = np.array([ 0.398 , 0.005, 0.65])
        self.max_away_frm_init_pose = 0.12
        self.desired_pose = np.array([ 0.33 , -0.005, 0.57])
        self.obs = np.concatenate((self.init_array, self.desired_pose))
        self.mu, self.sigma = 0, 0.001
        
        print("in the env init")
        
    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array) 
        """
        self.obs[0] = self.init_array[0]
        self.obs[1] = self.init_array[1]
        self.obs[2] = self.init_array[2]
        self.obs[3] = self.desired_pose[0]
        self.obs[4] = self.desired_pose[1]
        self.obs[5] = self.desired_pose[2]
        #print("Resetting env")
        return self.obs
    
    
    
    def step(self, action):
        old_obs = copy.deepcopy(self.obs)
        
        up_obs = copy.deepcopy(self.obs)[:3]
        
        
        up_obs[0] += action[0] * self.position_delta
        up_obs[1] += action[1] * self.position_delta
        up_obs[2] += action[2] * self.position_delta 
             
                
        up_obs = np.clip(up_obs, self.init_array-self.max_away_frm_init_pose,\
                              self.init_array+self.max_away_frm_init_pose)
        
        up_obs = up_obs + np.random.normal(self.mu, self.sigma, up_obs.shape) #added noise
        
        self.obs[:3] = copy.deepcopy(up_obs)
        

        dist = np.linalg.norm(self.desired_pose - self.obs[:3])
        
        #print("Dist from goal is", dist, "Last action is", action, "obs are", self.obs, "desired_pose", self.desired_pose  )
            
        done = np.all(np.isclose(self.desired_pose,  self.obs[:3], atol=self.tolerence))
        
        if done:
            reward = 50
        else:
            reward = -1
            
        info = {}
            
        return self.obs, reward, done, info

    
    def render(self, mode='human', close=False):
        # Render the environment to the screen
        pass
    