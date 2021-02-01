# Franka Imeka Panda robot gym environment for Reinforcement Learning  [Panda + ROS + RL + RVIZ]

**No hassle package for training your robot!**

It is the environment to train the Franka Imeka Panda robot to reach a particular position with a gym interface. This code is sim2real compatible.

[OpenAI ROS](http://wiki.ros.org/openai_ros) provides a detailed tutorial on how to create gym-style environments and use Openai's Baseline algorithms with them using the Gazebo simulator. 

Unfortunately, the Panda robot does not have an official Gazebo simulator. In package usages, RVIZ environment, provided with the official [MoveIT Tutorials](http://docs.ros.org/en/melodic/api/moveit_tutorials/html/doc/quickstart_in_rviz/quickstart_in_rviz_tutorial.html).

This package assumes that you already have [Franka-ROS](https://frankaemika.github.io/docs/franka_ros.html) package installed with ROS melodic or kinetic and that you have [moveit tutorials](http://docs.ros.org/en/melodic/api/moveit_tutorials/html/doc/quickstart_in_rviz/quickstart_in_rviz_tutorial.html) installed.

Simply put the package in the src directory of your ROS workspace and say catkin_make or catkin build.

Later code simply runs by using two commands.

Start panda robot in RVIZ.

In one terminal run:
>> roslaunch panda_moveit_config demo.launch 

In another terminal run:
>> roslaunch panda_openai_ros panda_deepq.launch

 You should be able to see the average return of the trained robot by echoing the following topic.
 >>rostopic echo /openai/reward
 
 Voila! You are ready!
