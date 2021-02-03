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
#https://bitbucket.org/theconstructcore/theconstruct_msgs/src/master/msg/RLExperimentInfo.msg
from panda_env.msg import RLExperimentInfo
import tf as tff

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


class PandaEnv(gym.Env):

    def __init__(self, ros_ws_abspath):
        rospy.logdebug("Entered Panda Env")
                
        self.controllers_list = []

        self.robot_name_space = ""
        self.reset_controls = False   
        self.seed()

        # Set up ROS related variables
        self.episode_num = 0
        self.cumulated_episode_reward = 0
        self.reward_pub = rospy.Publisher('/openai/reward', RLExperimentInfo, queue_size=1)     
                
        self.JOINT_STATES_SUBSCRIBER = '/joint_states'
        self.join_names = ["panda_joint1",
                          "panda_joint2",
                          "panda_joint3",
                          "panda_joint4",
                          "panda_joint5",
                          "panda_joint6", "panda_joint7"]
                
        self.joint_states_sub = rospy.Subscriber(self.JOINT_STATES_SUBSCRIBER, JointState, self.joints_callback)
        
        self.joints = JointState()
        
        # Start Services
        
        self.base_link = 'panda_link0' #'panda_link0' 
        self.ee_link = 'panda_EE'
        
        self.move_panda_object = MovePanda(self.base_link, self.ee_link)
        
        self.gripper_orientation = self.move_panda_object.get_gripper_quat()
        
        # Wait until it has reached its Sturtup Position
        self.wait_panda_ready()
        
    # PandaEnv virtual methods
    # ----------------------------

    def _check_all_sensors_ready(self):
        self._check_joint_states_ready()
        
        rospy.logdebug("ALL SENSORS READY")
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def step(self, action):
        """
        Function executed each time step.
        Here we get the action execute it in a time step and retrieve the
        observations generated by that action.
        :param action:
        :return: obs, reward, done, info
        """

        """
        Here we should convert the action num to movement action, execute the action in the
        simulation and get the observations result of performing that action.
        """
        rospy.logdebug("START STEP OpenAIROS")

        self._set_action(action)
        obs = self._get_obs()
        done = self._is_done(obs)
        info = {'is_success' : done}
        reward = self._compute_reward(obs, done)
        self.cumulated_episode_reward += reward
        #print("Total reward is", self.cumulated_episode_reward, "done is", done, obs, self.desired_position)

        rospy.logdebug("END STEP OpenAIROS")

        return obs, reward, done, info

    def _check_joint_states_ready(self):
        self.joints = None
        while self.joints is None and not rospy.is_shutdown():
            try:
                self.joints = rospy.wait_for_message(self.JOINT_STATES_SUBSCRIBER, JointState, timeout=1.0)
                rospy.logdebug("Current "+str(self.JOINT_STATES_SUBSCRIBER)+" READY=>" + str(self.joints))

            except:
                rospy.logerr("Current "+str(self.JOINT_STATES_SUBSCRIBER)+" not ready yet, retrying....")
        return self.joints
    
    def reset(self):
        rospy.loginfo("Reseting robot pose")
        rospy.logdebug("Reseting RobotGazeboEnvironment")
        self._reset_sim()
        self._init_env_variables()
        self._update_episode()
        obs = self._get_obs()
        rospy.logdebug("END Reseting RobotGazeboEnvironment")
        return obs

    def close(self):
        """
        Function executed when closing the environment.
        Use it for closing GUIS and other systems that need closing.
        :return:
        """
        rospy.logdebug("Closing RobotGazeboEnvironment")
        rospy.signal_shutdown("Closing RobotGazeboEnvironment")

    def _update_episode(self):
        """
        Publishes the cumulated reward of the episode and
        increases the episode number by one.
        :return:
        """
        rospy.logwarn("PUBLISHING REWARD...")
        self._publish_reward_topic(
                                    self.cumulated_episode_reward,
                                    self.episode_num
                                    )
        rospy.logwarn("PUBLISHING REWARD...DONE="+str(self.cumulated_episode_reward)+",EP="+str(self.episode_num))

        self.episode_num += 1
        self.cumulated_episode_reward = 0


    def _publish_reward_topic(self, reward, episode_number=1):
        """
        This function publishes the given reward in the reward topic for
        easy access from ROS infrastructure.
        :param reward:
        :param episode_number:
        :return:
        """
        reward_msg = RLExperimentInfo()
        reward_msg.episode_number = episode_number
        reward_msg.episode_reward = reward
        self.reward_pub.publish(reward_msg)

    # Extension methods
    # ----------------------------

    def _reset_sim(self):
        """Resets a simulation
        """
        rospy.loginfo("RESET SIM START")
        self._set_init_pose()
        rospy.logdebug("RESET SIM END")
        #print("RESET SIM END")
        return True
    
    def joints_callback(self, data):
        self.joints = data

    def get_joints(self):
        return self.joints
        
    def get_joint_names(self):
        return self.joints.name

    def set_trajectory_ee(self, action):
        """
        Sets the Pose of the EndEffector based on the action variable.
        The action variable contains the position and orientation of the EndEffector.
        See create_action
        """
        # Set up a trajectory message to publish.
        ee_target = geometry_msgs.msg.Pose()
        
        ee_target.orientation.x= self.gripper_orientation[0]
        ee_target.orientation.y= self.gripper_orientation[1]
        ee_target.orientation.z= self.gripper_orientation[2]
        ee_target.orientation.w= self.gripper_orientation[3]
        
        ee_target.position.x = action[0]
        ee_target.position.y = action[1]
        ee_target.position.z = action[2]
        
        rospy.logdebug("Set Trajectory EE...START...POSITION="+str(ee_target.position))
        result = self.move_panda_object.ee_traj(ee_target)
        rospy.logdebug("Set Trajectory EE...END...RESULT="+str(result))
        
        return result
        
    def set_initial_pose(self, initial_pos):
        initial_pos = np.array([[initial_pos['x'], initial_pos['y'], initial_pos['z']]])
    
        self.move_panda_object.set_pose(initial_pos)
        
        return True
        
    def create_action(self,position,orientation):
        """
        position = [x,y,z]
        orientation= [x,y,z,w]
        """
        
        gripper_target = np.array(position)
        gripper_rotation = np.array(orientation)
        #print("gripper_rotation", gripper_target, gripper_rotation)
        action = np.concatenate([gripper_target.flatten(), gripper_rotation.flatten()])
        
        return action
        
    def create_joints_dict(self,joints_positions):
        """
        Based on the Order of the positions, they will be assigned to its joint name
        names_in_order:
          joint0: 0.0
          joint1: 0.0
          joint2: 0.0
          joint3: -1.5
          joint4: 0.0
          joint5: 1.5
          joint6: 0.0
        """
        
        assert len(joints_positions) == len(self.join_names), "Wrong number of joints, there should be "+str(len(self.join_names))
        joints_dict = dict(zip(self.join_names,joints_positions))
        
        return joints_dict
        
    def get_ee_pose(self):
        """
        Returns geometry_msgs/PoseStamped
            std_msgs/Header header
              uint32 seq
              time stamp
              string frame_id
            geometry_msgs/Pose pose
              geometry_msgs/Point position
                float64 x
                float64 y
                float64 z
              geometry_msgs/Quaternion orientation
                float64 x
                float64 y
                float64 z
                float64 w
        """
        #self.gazebo.unpauseSim()
        gripper_pose = self.move_panda_object.ee_pose()
        #self.gazebo.pauseSim()
        
        return gripper_pose
        
    def get_ee_rpy(self):
        
        gripper_rpy = self.move_panda_object.ee_rpy()
        
        return gripper_rpy
        
    def wait_panda_ready(self):

        print("WAITING...for panda to get ready")
        import time
        for i in range(2):
            current_joints = self.get_joints()
            joint_pos = current_joints.position
            print("JOINTS POS NOW="+str(joint_pos))
            print("WAITING..."+str(i))
            time.sleep(1.0)
            
        print("WAITING...DONE")      
    
    # ParticularEnv methods
    # ----------------------------

    def _init_env_variables(self):
        """Inits variables needed to be initialised each time we reset at the start
        of an episode.
        """
        raise NotImplementedError()

    def _compute_reward(self, observations, done):
        """Calculates the reward to give based on the observations given.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _get_obs(self):
        raise NotImplementedError()

    def _is_done(self, observations):
        """Checks if episode done based on observations given.
        """
        raise NotImplementedError()
    
    def get_gripper_orientation(self):
        self.gripper_orientation = self.move_panda_object.get_gripper_quat()
        return self.gripper_orientation
        
    def getForce(self):
        return self.move_panda_object.force
        
        
class MovePanda:
    
    def __init__(self, base_link, ee_link):
        
        self.listener = tff.TransformListener()
         #'panda_link8' ##TODO check this
        self.goal_tolerence = 0.03
        #self.max_away_frm_init_pose = 0.2 #rospy.get_param('/panda/max_away_frm_init_pose')
        self.goal_offset = 0.02 #self.tolerence #0.03
        self.axis_offset = 0.02 #self.tolerence #0.03
        self.cart_pose_array = np.ones((1,3)) * self.axis_offset
        self.goal_publisher = rospy.Publisher("/equilibrium_pose", PoseStamped, queue_size =10)
        self.rate = rospy.Rate(20) # 10hz        
        rospy.Subscriber('/franka_state_controller/F_ext',
                                                    WrenchStamped, self.getForceCB, queue_size=1, tcp_nodelay=True)
        
        self.force = None
        self.base_link = base_link
        self.ee_link = ee_link
        
        
    def getForceCB(self, msg):
        self.force = np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z, msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z])

    
    def getTFdata(self):
        try:
            (trans,rot) = self.listener.lookupTransform(self.base_link, self.ee_link, rospy.Time(0) )
            curr_pose = np.array(sum([trans, rot], [])).reshape(1,-1)
            return curr_pose 
        except(tff.LookupException, tff.ConnectivityException, tff.ExtrapolationException):
            return []
            pass       
        
    def goal_pose_in_cart(self, goal_pos):
        curr_pose = []
        #print('getting tf data')
        '''
        goal_pos = np.array([[goal_pose.pose.position.x, goal_pose.pose.position.y, goal_pose.pose.position.z,
                              goal_pose.pose.orientation.x, goal_pose.pose.orientation.y, goal_pose.pose.orientation.z,
                              goal_pose.pose.orientation.w]])
        '''
        while len(curr_pose) == 0:
            curr_pose = self.getTFdata()
        print('Got tf data',goal_pos, curr_pose )
        goal_pose_t = goal_pos - curr_pose[0, :3]
        
        #print( "actual goal pose: goal_pose_t", goal_pose_t)
        goal_pose_t_offset = copy.deepcopy(goal_pose_t)
        
        for i in range(goal_pose_t.shape[1]):
            if goal_pose_t[0][i] <= self.axis_offset and goal_pose_t[0][i] >= -self.axis_offset:
                pass
            else:
                goal_pose_t_offset[0][i] = np.copysign(self.axis_offset, goal_pose_t[0][i])
                      
        #print( "Later goal pose: goal_pose_t", goal_pose_t_offset)
        
        #if True: #np.linalg.norm(goal_pose_t, 1) > self.goal_offset:
        goal = PoseStamped()
        goal.header.seq = 1
        goal.header.stamp = rospy.Time.now()
        goal.header.frame_id = ''
        goal.pose.position.x = curr_pose[0][0] + goal_pose_t_offset[0][0] 
        goal.pose.position.y = curr_pose[0][1] + goal_pose_t_offset[0][1] 
        goal.pose.position.z = curr_pose[0][2] + goal_pose_t_offset[0][2]
        
        gripper_orientation = self.get_gripper_quat()
        goal.pose.orientation.x = gripper_orientation[0]
        goal.pose.orientation.y = gripper_orientation[1]
        goal.pose.orientation.z = gripper_orientation[2]
        goal.pose.orientation.w = gripper_orientation[3]
        #print("Publishing goal")
        self.goal_publisher.publish(goal)
        
        return self.goal_reached(np.array([[goal.pose.position.x, goal.pose.position.y, goal.pose.position.z,
                                       goal.pose.orientation.x, goal.pose.orientation.y, goal.pose.orientation.z, 
                                       goal.pose.orientation.w]]).reshape(1,-1))
            
            
    def goal_reached(self, goal_pose, tolerence_pos=0.02, tolerence_quat=0.01): #padmaja may be change this tolerance
        curr_pose = []
        #print('getting tf data in pub goal_reached')
        while len(curr_pose) == 0:
            curr_pose = self.getTFdata()
        #print("dist is", (curr_pose[0,:3]- goal_pose[0,:3]), curr_pose[0,:3], goal_pose[0,:3], tolerence_pos )
        if np.allclose(curr_pose[0,:3], goal_pose[0,:3], atol=self.goal_tolerence):#and np.allclose(curr_pose[0,3:], goal_pose[0,3:], tolerence_quat):
            return True
        return False  
    
    def ee_traj(self, goal_pos, timesteps=1):
        goal_pose = np.array([[goal_pos.position.x, goal_pos.position.y, goal_pos.position.z]])
        for i in range(timesteps):
            self.goal_pose_in_cart(goal_pose)   
            #print("Tring to reach the goal pose in ee_traj")  
            self.rate.sleep()    
        return True
        
    def set_pose(self, goal_pose):
        reached =  self.goal_pose_in_cart(goal_pose)
        while not reached:
            reached = self.goal_pose_in_cart(goal_pose)
            #print("Tring to reach the goal pose")
            self.rate.sleep()
        return reached
            
    def execute_trajectory(self):
        self.goal_pose_in_cart(self, goal_pos)   
        #print("Tring to reach the goal pose in execute_trajectory")     
        return True

    def ee_pose(self):
        curr_pose = []
        #print('getting tf data')
        while len(curr_pose) == 0:
            curr_pose = self.getTFdata()
        gripper_pose = PoseStamped()
   
        gripper_pose.pose.position.x = curr_pose[0][0] 
        gripper_pose.pose.position.y = curr_pose[0][1] 
        gripper_pose.pose.position.z = curr_pose[0][2]
        
        gripper_pose.pose.orientation.x = curr_pose[0][3]
        gripper_pose.pose.orientation.y = curr_pose[0][4]
        gripper_pose.pose.orientation.z = curr_pose[0][5]
        gripper_pose.pose.orientation.w = curr_pose[0][6]
        return gripper_pose
    
    def get_gripper_quat(self):
        curr_pose = []
        #print('getting tf data')
        while len(curr_pose) == 0:
            curr_pose = self.getTFdata()
        return curr_pose[0, 3:]
        