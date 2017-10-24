# Reinforcement Learning School for Baxter Robots

Requirements:
1. Numpy and Matplotlib

2. Baxter arm simulator (ROS) :
  * https://bitbucket.org/u2isir/arm_scenario_simulator

3. Pytorch :
  * http://pytorch.org/
  * http://pytorch.org/docs/


Basically to setup this repo:
(You need your 'catkin_ws' to be in your home ~)

```
git clone https://github.com/Mathieu-Seurin/3D_Baxter_representation_learning.git
```


Running the experiment episodes:
1. 
```
./process.sh
```
Then gazebo should open (you have to wait a little bit). Wait until you see
```
# [ INFO] [1508766121.393632990, 114.792000000]: Simulator is loaded and started successfully
# [ INFO] [1508766121.395720127, 114.802000000]: Robot is disabled
# [ INFO] [1508766121.395810571, 114.802000000]: Gravity compensation was turned off
```
2. Run in separate terminal window
rosrun arm_scenario_simulator spawn_objects_example

3. Then, in a new terminal (don't try to interrupt the other one)
```
python wholeProcess.py
```

WARNING : 
--------

Because of threading and ros ctrl-c usually doesn't work (or the error raised is a 'BaxterProblem' not 'KeyboardInterrupt')
If you want to interrupt the program do :
ctrl+\\
(backslash)
