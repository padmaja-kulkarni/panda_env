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
from panda_robot_impedance_env import PandaEnv


import copy
import rospy
import threading
#import quaternion
import numpy as np
from geometry_msgs.msg import Point
from visualization_msgs.msg import *
from interactive_markers.interactive_marker_server import *
#from franka_core_msgs.msg import EndPointState, JointCommand, RobotState
from geometry_msgs.msg import Pose, PoseStamped
from std_msgs.msg import String
from geometry_msgs.msg import WrenchStamped


def LoadYamlFileParamsTest(rospackage_name, rel_path_from_package_to_file, yaml_file_name):

    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path(rospackage_name)
    config_dir = os.path.join(pkg_path, rel_path_from_package_to_file) 
    path_config_file = os.path.join(config_dir, yaml_file_name)
    
    paramlist=rosparam.load_file(path_config_file)
    
    for params, ns in paramlist:
        rosparam.upload_params(ns,params)

class PandaImpedanceEnv(PandaEnv, utils.EzPickle):
    
    def __init__(self):
        
        ros_ws_abspath = rospy.get_param("/panda/ros_ws_abspath", '/home/padmaja/ros/')
        assert ros_ws_abspath is not None, "You forgot to set ros_ws_abspath in your \
        yaml file of your main RL script. Set ros_ws_abspath: \'YOUR/SIM_WS/PATH\'"
        assert os.path.exists(ros_ws_abspath), "The Simulation ROS Workspace path " + ros_ws_abspath + \
                                               " DOESNT exist, execute: mkdir -p " + ros_ws_abspath + \
                                               "/src;cd " + ros_ws_abspath + ";catkin_make"
        
        
        
        
        # Load Params from the desired Yaml file relative to this TaskEnvironment
        LoadYamlFileParamsTest(rospackage_name="panda_env",
                               rel_path_from_package_to_file="config",
                               yaml_file_name="panda_impedence_env.yaml")
        rospy.logdebug("Entered PandaTestEnv Env from reacher node")
        
        self.get_params()
        
        #panda_env.PandaEnv.__init__(self)
        super(PandaImpedanceEnv, self).__init__(ros_ws_abspath, self.goal_tolerence, self.position_delta, self.base_link, self.ee_link, self.gripper_orientation)
        
        self.gripper_rotation = self.get_gripper_orientation() #[0.924, -0.382, 0.000, 0.000]

        #self.action_space = spaces.Box(low=-0.04, high=0.04, shape=(self.n_actions,), dtype=np.float32) #padmaja might need to change it to cont.
        #self.n_actions = n_actions # Number of force values the agent can choose
        low_act = np.array([-self.position_delta]*self.n_actions)
        high_act = np.array([self.position_delta]*self.n_actions)
        self.action_space = spaces.Box(low=low_act, high=high_act)
        #self.action_space = spaces.Discrete(self.n_actions)
        
        observations_high_range = np.inf * np.ones(self.n_observations)
        observations_low_range = -observations_high_range
        
        self.observation_space = spaces.Box(observations_low_range, observations_high_range)
        
        self.curr_gripper_pose = None
                
        #print("\n\n\nShape is of obs and action", observations_high_range.shape, self.action_space.shape)
        
        

        
    def get_params(self):
        #get configuration parameters
        
        self.n_actions = rospy.get_param('/panda/n_actions')
        self.n_observations = rospy.get_param('/panda/n_observations')
        self.position_ee_max = rospy.get_param('/panda/position_ee_max')
        self.position_ee_min = rospy.get_param('/panda/position_ee_min')
        
        #init_pos = rospy.get_param('/panda/init_pos')
        #self.init_pose_array = np.array([self.init_pos["x"],self.init_pos["y"],self.init_pos["z"]])
        
        self.setup_ee_pos = rospy.get_param('/panda/setup_ee_pos')
        self.setup_ee_pos_array = np.array([self.setup_ee_pos["x"],self.setup_ee_pos["y"],self.setup_ee_pos["z"]])
        
        self.goal_ee_pos = rospy.get_param('/panda/goal_ee_pos')
        
        self.pid_goal = rospy.get_param('/panda/pid_goal_ee_pos')
        
        self.pid_goal_pose = np.array([self.pid_goal["x"],self.pid_goal["y"],self.pid_goal["z"]])
        
        self.position_delta = rospy.get_param('/panda/position_delta')
        self.step_punishment = rospy.get_param('/panda/step_punishment')
        self.closer_reward = rospy.get_param('/panda/closer_reward')
        self.impossible_movement_punishement = rospy.get_param('/panda/impossible_movement_punishement')
        self.reached_goal_reward = rospy.get_param('/panda/reached_goal_reward')
        
        #self.max_distance = rospy.get_param('/panda/max_distance')
        
        self.goal_tolerence = rospy.get_param('/panda/goal_tolerence') #0.03
        #self.axis_tolerence = rospy.get_param('/panda/axis_tolerence') #max distance that a robot can go in each direction
        self.max_away_frm_init = rospy.get_param('/panda/max_away_frm_init_pose')  #0.2
        self.max_away_frm_init_pose = np.array([self.max_away_frm_init["x"], self.max_away_frm_init["y"], self.max_away_frm_init["z"]])
        
        self.gripp_ori =  rospy.get_param('/panda/gripper_orientation')
        self.gripper_orientation = np.array([self.gripp_ori["x"], self.gripp_ori["y"], self.gripp_ori["z"], self.gripp_ori["w"]])
        
        self.desired_position = [self.goal_ee_pos["x"], self.goal_ee_pos["y"], self.goal_ee_pos["z"]]
        
        self.base_link = rospy.get_param('panda/base_link')
        self.ee_link = rospy.get_param('panda/ee_link')
        self.filter_alpha = rospy.get_param('panda/filter_alpha', 0.8)
        self.step_count = 0
        
        
    
    
    def _set_init_pose(self):
        """Sets the Robot in its init pose
        The Simulation will be unpaused for this purpose.
        """
        # Check because it seems its not being used
        #rospy.logdebug("Init Pos:")
        #rospy.logdebug(self.init_pos)

        # Init Joint Pose
        rospy.logdebug("Moving To SETUP Joints ")
        grip_pos = self.get_ee_pose()
        self.curr_gripper_pose = np.array([grip_pos.pose.position.x, grip_pos.pose.position.y, grip_pos.pose.position.z])
        """
        Openrates in an infinite while loop until in initial end effector pose is reached.
        """
        self.movement_result = self.set_initial_pose(self.setup_ee_pos) #padmaja
        
        #print("Gripper initial and setup poses are", self.curr_gripper_pose, self.setup_ee_pos)
        """
        Getting  the reached initial robot pose
        """
        grip_pos = self.get_ee_pose()
        self.last_gripper_target = np.array([grip_pos.pose.position.x, grip_pos.pose.position.y, grip_pos.pose.position.z])
        self.current_dist_from_des_pos_ee = self.calculate_distance_between(self.desired_position,self.last_gripper_target)
        
        #print("Init pose set")
        


    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        The simulation will be paused, therefore all the data retrieved has to be 
        from a system that doesnt need the simulation running, like variables where the 
        callbackas have stored last know sesnor data.
        :return:
        """
        rospy.logdebug("Init Env Variables...")
        rospy.logdebug("Init Env Variables...END")
        self.step_count = 0
    

    def _set_action(self, action):
        
        
        gripper_target = copy.deepcopy(self.curr_gripper_pose)
        
        #print(" self.pid_goal_pose")
        #action[0] = 0.0
        #action[1] = 0.0
        #action[2] = 0.0
                
        pid_pose_diff = self.pid_goal_pose - gripper_target
        
        pid_pose_clipped = np.clip(pid_pose_diff, -self.position_delta, self.position_delta)  
        
        #print(pid_pose_clipped, pid_pose_diff, gripper_target) 
        decay = (100.0 - self.step_count)/100.0 #0.5 #np.exp(-self.step_count/1.) #np.exp(-self.step_count/1.)
         

        gripper_target[0] += action[0] * (1-decay) + pid_pose_clipped[0] * decay
        gripper_target[1] += action[1]* (1-decay) + pid_pose_clipped[1] * decay
        gripper_target[2] += action[2]* (1-decay) + pid_pose_clipped[2] * decay
        
        """
        gripper_target[0] += action[0]/3.  + pid_pose_clipped[0]
        gripper_target[1] += action[1]/3.  + pid_pose_clipped[1]
        gripper_target[2] += action[2]/3.  + pid_pose_clipped[2]
        """
    
        
        #gripper_target = gripper_target + np.random.normal(self.mu, self.sigma, up_obs.shape)
        
        gripper_target = np.clip(gripper_target, self.setup_ee_pos_array-self.max_away_frm_init_pose,\
                              self.setup_ee_pos_array+self.max_away_frm_init_pose)
        
        
        """
        Filter
        """
        
        if self.step_count == 0:
            self.last_action = copy.deepcopy(gripper_target)
        else:
            gripper_target = self.filter_alpha * gripper_target + (1. - self.filter_alpha) * self.last_action
            self.last_action = copy.deepcopy(gripper_target)
        
        
        self.step_count += 1
        #print("Action target is", action)
        #print("Gripper target is", gripper_target)
        #print("pid_pose_clipped target is", pid_pose_clipped)        
        
        # Apply action to simulation.
        action_end_effector = self.create_action(gripper_target, self.get_gripper_orientation())
        self.movement_result = self.set_trajectory_ee(action_end_effector)
        if self.movement_result:
            # If the End Effector Positioning was succesfull, we replace the last one with the new one.
            self.last_gripper_target = copy.deepcopy(gripper_target)
        else:
            rospy.logerr("Impossible End Effector Position...."+str(gripper_target))
        
        #rospy.logwarn("END Set Action ==>"+str(action)+", NAME="+str(self.last_action))

    def _get_obs(self):
        """
        It returns the Position of the TCP/EndEffector as observation.
        And the distance from the desired point
        Orientation for the moment is not considered
        """
        
        grip_pos = self.get_ee_pose()
        grip_pos_array = np.array([grip_pos.pose.position.x, grip_pos.pose.position.y, grip_pos.pose.position.z])
        
        force_array = self.getForce()
        #print("\n\n\nforce are:", force_array)
        if force_array.any() == None:
            force_array = np.zeros((6,1))
        assert not np.all(force_array == 0.)
        """
        ======================================================================================================
        First three sould always be the gripper pose. It is used lated to determine curr_pose ####VVVIIIIIPPPPP
        =====================================================================================================
        
        """
        
        #print("grip_pos_array,", grip_pos_array,  force_array)
        obs = np.concatenate((grip_pos_array.flatten(), force_array.flatten()))
        obs = np.concatenate((obs, self.pid_goal_pose))
        
        self.curr_gripper_pose = copy.deepcopy(grip_pos_array)
        
        #new_dist_from_des_pos_ee = self.calculate_distance_between(self.desired_position, grip_pos_array)
        
        #obs.append(new_dist_from_des_pos_ee)
        
        #print("Obs are:", obs)
        
        
        #print("\n\n\n\n\nn+++>>>>>>>>>>>>>grip_pos_array", np.array(grip_pos_array))
        return obs
        
    def _is_done(self, observations):
        
        """
        If the latest Action didnt succeed, it means that tha position asked was imposible therefore the episode must end.
        It will also end if it reaches its goal.
        """
        #print("\n\n\n observations", observations)
        current_pos = observations[:3]
        
        self.curr_gripper_pose = observations[:3]
        
        done = self.calculate_if_done(self.movement_result,self.desired_position,current_pos)
        if done:
            return 1
        else:
            return 0
        
    def _compute_reward(self, observations, done):

        """
        We punish each step that it passes without achieveing the goal.
        Punishes differently if it reached a position that is imposible to move to.
        Rewards getting to a position close to the goal.
        """
        #print("\n\n\n\n\n Current pose", observations)
        current_pos = observations[:3]
        
        new_dist_from_des_pos_ee = self.calculate_distance_between(current_pos, self.desired_position)
        
        reward = self.calculate_reward(self.movement_result, self.desired_position, current_pos, new_dist_from_des_pos_ee)
        rospy.logwarn(">>>REWARD>>>"+str(reward))
        
        return reward
    
    def calculate_reward(self, movement_result, desired_position, current_pos, new_dist_from_des_pos_ee):
        """
        It calculated whather it has finished or nota and how much reward to give
        """

        if movement_result:
            position_similar = np.all(np.isclose(desired_position, current_pos, atol=self.goal_tolerence))
            
            # Calculating Distance
            rospy.logwarn("desired_position="+str(desired_position))
            rospy.logwarn("current_pos="+str(current_pos))
            rospy.logwarn("self.current_dist_from_des_pos_ee="+str(self.current_dist_from_des_pos_ee))
            rospy.logwarn("new_dist_from_des_pos_ee="+str(new_dist_from_des_pos_ee))
            
            delta_dist = new_dist_from_des_pos_ee - self.current_dist_from_des_pos_ee
            if position_similar:
                reward = self.reached_goal_reward
                rospy.logwarn("Reached a Desired Position!")
            else:
                if delta_dist < 0:
                    reward = self.closer_reward
                    rospy.logwarn("CLOSER To Desired Position!="+str(delta_dist))
                else:
                    reward = self.step_punishment
                    rospy.logwarn("FURTHER FROM Desired Position!"+str(delta_dist))
            
        else:
            reward = self.impossible_movement_punishement
            rospy.logwarn("Reached a TCP position not reachable")
            
        # We update the distance
        self.current_dist_from_des_pos_ee = new_dist_from_des_pos_ee
        rospy.logdebug("Updated Distance from GOAL=="+str(self.current_dist_from_des_pos_ee))
            
        return reward
            
        

    def calculate_if_done(self, movement_result,desired_position,current_pos):
        """
        It calculated whather it has finished or not
        """
        done = False

        dist = self.calculate_distance_between(desired_position, current_pos)
        
        #print("Dist from goal is", dist, "Last action is", self.last_action )
            
        return np.all(np.isclose(desired_position, current_pos, atol=self.goal_tolerence))
        
    def calculate_distance_between(self,v1,v2):
        """
        Calculated the Euclidian distance between two vectors given as python lists.
        """
        dist = np.linalg.norm(np.array(v1)-np.array(v2))#np.fabs(v1[2]-v2[2]) #np.linalg.norm(np.array(v1)-np.array(v2)) #padmaja
        return dist