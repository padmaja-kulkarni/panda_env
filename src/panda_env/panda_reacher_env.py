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
from panda_robot_env import PandaEnv





def LoadYamlFileParamsTest(rospackage_name, rel_path_from_package_to_file, yaml_file_name):

    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path(rospackage_name)
    config_dir = os.path.join(pkg_path, rel_path_from_package_to_file) 
    path_config_file = os.path.join(config_dir, yaml_file_name)
    
    paramlist=rosparam.load_file(path_config_file)
    
    for params, ns in paramlist:
        rosparam.upload_params(ns,params)

class PandaTestEnv(PandaEnv, utils.EzPickle):
    def __init__(self):
        
        ros_ws_abspath = rospy.get_param("/panda/ros_ws_abspath", None)
        assert ros_ws_abspath is not None, "You forgot to set ros_ws_abspath in your yaml file of your main RL script. Set ros_ws_abspath: \'YOUR/SIM_WS/PATH\'"
        assert os.path.exists(ros_ws_abspath), "The Simulation ROS Workspace path " + ros_ws_abspath + \
                                               " DOESNT exist, execute: mkdir -p " + ros_ws_abspath + \
                                               "/src;cd " + ros_ws_abspath + ";catkin_make"
        
        
        # Load Params from the desired Yaml file relative to this TaskEnvironment
        LoadYamlFileParamsTest(rospackage_name="panda_env",
                               rel_path_from_package_to_file="config",
                               yaml_file_name="panda_test.yaml")
        rospy.logdebug("Entered PandaTestEnv Env from reacher node")
        self.get_params()
        
        #panda_env.PandaEnv.__init__(self)


        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_actions,), dtype=np.float32) #padmaja might need to change it to cont.
        
        observations_high_range = np.array([self.position_ee_max]*self.n_observations)
        observations_low_range = np.array([self.position_ee_min]*self.n_observations)
        
        observations_high_dist = np.array([self.max_distance])
        observations_low_dist = np.array([0.0])
        
        high = np.concatenate([observations_high_range, observations_high_dist])
        low = np.concatenate([observations_low_range, observations_low_dist])
        
        self.observation_space = spaces.Box(low, high)
        
        self.tolerence = 0.05
        super(PandaTestEnv, self).__init__(ros_ws_abspath)
        
        

        
    def get_params(self):
        #get configuration parameters
        
        self.n_actions = rospy.get_param('/panda/n_actions')
        self.n_observations = rospy.get_param('/panda/n_observations')
        self.position_ee_max = rospy.get_param('/panda/position_ee_max')
        self.position_ee_min = rospy.get_param('/panda/position_ee_min')
        
        self.init_pos = rospy.get_param('/panda/init_pos')
        self.setup_ee_pos = rospy.get_param('/panda/setup_ee_pos')
        self.goal_ee_pos = rospy.get_param('/panda/goal_ee_pos')
        
        self.position_delta = rospy.get_param('/panda/position_delta')
        self.step_punishment = rospy.get_param('/panda/step_punishment')
        self.closer_reward = rospy.get_param('/panda/closer_reward')
        self.impossible_movement_punishement = rospy.get_param('/panda/impossible_movement_punishement')
        self.reached_goal_reward = rospy.get_param('/panda/reached_goal_reward')
        
        self.max_distance = rospy.get_param('/panda/max_distance')
        
        self.desired_position = [self.goal_ee_pos["x"], self.goal_ee_pos["y"], self.goal_ee_pos["z"]]
        self.gripper_rotation = [0.924, -0.382, 0.000, 0.000]
    
    
    def _set_init_pose(self):
        """Sets the Robot in its init pose
        The Simulation will be unpaused for this purpose.
        """
        # Check because it seems its not being used
        rospy.logdebug("Init Pos:")
        rospy.logdebug(self.init_pos)

        # Init Joint Pose
        rospy.logdebug("Moving To SETUP Joints ")
        self.movement_result = self.set_trajectory_joints(self.init_pos) #padmaja
        
        
        # We test the Desired Goal
        
        # INIT POSE
        print("Moving To TEST DESIRED GOAL Position from reacher node", self.desired_position)
        print("Moving To TEST DESIRED self.gripper_rotation Position from reacher node",self.gripper_rotation)
        action = self.create_action(self.desired_position,self.gripper_rotation)
        self.movement_result = self.set_trajectory_ee(action)
            
        #self.movement_result = True #padmaja
        if self.movement_result:
            # INIT POSE
            rospy.logdebug("Moving To SETUP Position ")
            self.last_gripper_target = [self.setup_ee_pos["x"],self.setup_ee_pos["y"],self.setup_ee_pos["z"]]
            action = self.create_action(self.last_gripper_target,self.gripper_rotation)
            self.movement_result = self.set_trajectory_ee(action)
            
            self.current_dist_from_des_pos_ee = self.calculate_distance_between(self.desired_position,self.last_gripper_target)
            rospy.logdebug("INIT DISTANCE FROM GOAL==>"+str(self.current_dist_from_des_pos_ee))
        else:
            #pass #padmaja
            assert False, "Desired GOAL EE is not possible"
        
        self.last_action = "INIT"
        
        rospy.logdebug("Init Pose Resultsare here ==>"+str(self.movement_result))

        self.movement_result #return True # #padmaja


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
    

    def _set_action(self, action):
        
        
        delta_gripper_target = [0.0]*len(self.last_gripper_target)
        
        #print("\n\n\nActions", action)
        # We convert action ints in increments/decrements of one of the axis XYZ
        #=======================================================================
        # if action == 0: # X+
        #     delta_gripper_target[0] += self.position_delta
        #     self.last_action = "X+"
        # elif action == 1: # X-
        #     delta_gripper_target[0] -= self.position_delta
        #     self.last_action = "X-"
        # elif action == 2: # Y+
        #     delta_gripper_target[1] += self.position_delta
        #     self.last_action = "Y+"
        # elif action == 3: # Y-
        #     delta_gripper_target[1] -= self.position_delta
        #     self.last_action = "Y-"
        # elif action == 4: #Z+
        #     delta_gripper_target[2] += self.position_delta
        #     self.last_action = "Z+"
        # elif action == 5: #Z-
        #     delta_gripper_target[2] -= self.position_delta
        #     self.last_action = "Z-"
        #=======================================================================
        
        delta_gripper_target[0] +=  action[0] * 0.01
        delta_gripper_target[1] +=  action[1] * 0.01
        delta_gripper_target[2] +=  action[2] * 0.01
        
        self.last_action = copy.deepcopy(action)
        
        
        
        
        gripper_target = copy.deepcopy(self.last_gripper_target)
        gripper_target[0] += delta_gripper_target[0]
        gripper_target[1] += delta_gripper_target[1]
        gripper_target[2] += delta_gripper_target[2]
        
        # Apply action to simulation.
        action_end_effector = self.create_action(gripper_target,self.gripper_rotation)
        self.movement_result = self.set_trajectory_ee(action_end_effector)
        if self.movement_result:
            # If the End Effector Positioning was succesfull, we replace the last one with the new one.
            self.last_gripper_target = copy.deepcopy(gripper_target)
        else:
            rospy.logerr("Impossible End Effector Position...."+str(gripper_target))
        
        rospy.logwarn("END Set Action ==>"+str(action)+", NAME="+str(self.last_action))

    def _get_obs(self):
        """
        It returns the Position of the TCP/EndEffector as observation.
        And the distance from the desired point
        Orientation for the moment is not considered
        """
        
        grip_pos = self.get_ee_pose()
        grip_pos_array = [grip_pos.pose.position.x, grip_pos.pose.position.y, grip_pos.pose.position.z]
        
        #obs = grip_pos_array
        
        #new_dist_from_des_pos_ee = self.calculate_distance_between(self.desired_position,grip_pos_array)
        
        #obs.append(new_dist_from_des_pos_ee)
        
        
        #print("\n\n\n\n\nn+++>>>>>>>>>>>>>grip_pos_array", np.array(grip_pos_array))

        
        obs_all= {
            'observation': np.array([grip_pos.pose.position.x, grip_pos.pose.position.y, grip_pos.pose.position.z]),
            'achieved_goal': np.array(grip_pos_array),
            'desired_goal': np.array(self.desired_position),
        }
        #print("OBSERVATIONS====>>>>>>>", (obs_all))
        return obs_all
        
    def _is_done(self, observations):
        
        """
        If the latest Action didnt succeed, it means that tha position asked was imposible therefore the episode must end.
        It will also end if it reaches its goal.
        """
        #print("\n\n\n observations", observations)
        current_pos = observations['achieved_goal']
        
        done = self.calculate_if_done(self.movement_result,self.desired_position,current_pos)
        
        return done
        
    def _compute_reward(self, observations, done):

        """
        We punish each step that it passes without achieveing the goal.
        Punishes differently if it reached a position that is imposible to move to.
        Rewards getting to a position close to the goal.
        """
        #print("\n\n\n\n\n Current pose", observations)
        current_pos = observations['achieved_goal']
        
        new_dist_from_des_pos_ee = self.calculate_distance_between(observations['desired_goal'],observations['achieved_goal'])
        
        reward = self.calculate_reward(self.movement_result, self.desired_position, current_pos, new_dist_from_des_pos_ee)
        rospy.logwarn(">>>REWARD>>>"+str(reward))
        
        return reward
    
    def compute_reward(self, observations, done, info):
        '''
        This reward is computed based on the samples replayed. Each observation here is a array of observations['achieved_goal']
        '''
        rewards = []
        for i in range(len(observations)):
            current_pos = observations[i]
            desired_position = self.desired_position
            position_similar = np.all(np.isclose(desired_position, current_pos, atol=self.tolerence))
            new_dist_from_des_pos_ee = self.calculate_distance_between(desired_position, current_pos)
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
            rewards.append(reward)
        return rewards
            
        

    def calculate_if_done(self, movement_result,desired_position,current_pos):
        """
        It calculated whather it has finished or not
        """
        done = False
        
        """
        if movement_result:
            
            position_similar = np.all(np.isclose(desired_position, current_pos, atol=1e-02))

            if position_similar:
                done = True
                rospy.logdebug("Reached a Desired Position!")
            else:
                done = False
        else:
            done = False
            rospy.logdebug("Reached a TCP position not reachable")
        """
            
        return np.all(np.isclose(desired_position, current_pos, atol=self.tolerence))
        
    
    def calculate_reward(self, movement_result, desired_position, current_pos, new_dist_from_des_pos_ee):
        """
        It calculated whather it has finished or nota and how much reward to give
        """

        if movement_result:
            position_similar = np.all(np.isclose(desired_position, current_pos, atol=self.tolerence))
            
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
        
    def calculate_distance_between(self,v1,v2):
        """
        Calculated the Euclidian distance between two vectors given as python lists.
        """
        dist = np.linalg.norm(np.array(v1)-np.array(v2))
        return dist
