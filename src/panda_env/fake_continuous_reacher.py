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

    def __init__(self, n_actions=3, n_observations=12, position_delta = 0.04, tol = 0.02, render=False):
        
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
        observations_high_range = np.inf * np.ones(self.n_observations)
        observations_low_range = -observations_high_range
        self.observation_space = spaces.Box(observations_low_range, observations_high_range)
        
        self.rendering = render
        self.position_delta = position_delta
        self.tolerence = tol
        self.init_array = np.array([0.600, -0.025, 0.464])
        self.max_away_frm_init_pose = 0.20
        self.desired_pose = np.array([ 0.768, -0.065, 0.464])
        self.pid_goal_pose = np.array([ 0.768, -0.025, 0.464 ])
        self.force_array = np.ones((6,)) * 0.05
        self.obs = np.concatenate((self.init_array, self.force_array))
        self.obs = np.concatenate((self.obs, self.pid_goal_pose))
        self.mu, self.sigma = 0, 0.001
        self.step_count = 0
        self.filter_alpha = 0.8
        
        #self.last_actions = np.zeros((self.filter_window, self.n_actions))
        
        print("in the env init")
        
    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array) 
        """
        self.obs[0] = self.init_array[0]
        self.obs[1] = self.init_array[1]
        self.obs[2] = self.init_array[2]
        self.step_count = 0
        #print("Resetting env")
        return self.obs
    
    def get_decay(self):
        pass
    
    
    
    def step(self, action):
        old_obs = copy.deepcopy(self.obs)
        
        up_obs = copy.deepcopy(self.obs)[:3]
        
        pid_pose_diff = self.pid_goal_pose - up_obs
        
        pid_pose_clipped = np.clip(pid_pose_diff, -self.position_delta, self.position_delta)  
        
        #print(pid_pose_clipped, pid_pose_diff)  
        decay = np.exp(-self.step_count/30.)#(50.0 - self.step_count)/50.0
        
        
        #print("Decay is", decay, self.step_count)
        
        
        
        up_obs[0] += action[0] * (1-decay) + pid_pose_clipped[0] * decay
        up_obs[1] += action[1]* (1-decay) + pid_pose_clipped[1] * decay
        up_obs[2] += action[2]* (1-decay) + pid_pose_clipped[2] * decay
        
        """
        up_obs[0] += action[0]  + pid_pose_clipped[0]
        up_obs[1] += action[1]+ pid_pose_clipped[1]
        up_obs[2] += action[2] + pid_pose_clipped[2] 
        """    
                
        up_obs = np.clip(up_obs, self.init_array-self.max_away_frm_init_pose,\
                              self.init_array+self.max_away_frm_init_pose)
        
        up_obs = up_obs + np.random.normal(self.mu, self.sigma, up_obs.shape) #added noise
        
        """
        Filter
        """
        #print("\n\n obs before", up_obs)
        if self.step_count == 0:
            self.last_action = copy.deepcopy(up_obs)
        else:
            up_obs = self.filter_alpha * up_obs + (1. - self.filter_alpha) * self.last_action
            self.last_action = copy.deepcopy(up_obs)
            
        #print("After, up_obs", up_obs)
        
        self.step_count += 1
        
        self.obs[:3] = copy.deepcopy(up_obs)
        

        dist = np.linalg.norm(self.desired_pose - self.obs[:3])
        
        #print("Dist from goal is", dist, "Last action is", action, "obs are", self.obs, "desired_pose", self.desired_pose  )
            
        done = np.all(np.isclose(self.desired_pose,  self.obs[:3], atol=self.tolerence))
        
        if done:
            #print(" self.desired_pose,  self.obs[:3] ", self.desired_pose,  self.obs[:3])
            reward = 50
        else:
            reward = -1
            
        info = {}
            
        return self.obs, reward, done, info

    
    def render(self, mode='human', close=False):
        # Render the environment to the screen
        pass
    