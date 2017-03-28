# enstage

Welcome.

At the moment, the project is not nearly to be completed, so it can change from time to time.

This repo is linked to this one : https://github.com/TLESORT/Baxter_Learning/commits/master

You need :

Baxter arm simulator (ROS) : 
https://bitbucket.org/u2isir/arm_scenario_simulator

Pytorch :
http://pytorch.org/
http://pytorch.org/docs/

Numpy and Matplotlib

Basically to setup this repo :
(You need your 'catkin_ws' to be in your home ~)

> git clone https://github.com/Mathieu-Seurin/enstage.git;

> cd enstage;

> git clone https://github.com/TLESORT/Baxter_Learning.git;

> ./process.sh;

Then gazebo should open (you have to wait a little bit)

Then, in a new terminal (don't try to interrupt the other one)
> python wholeProcess.py


WARNING : 
--------

Because of threading and ros ctrl-c usually doesn't work (or the error raised is a 'BaxterProblem' not 'KeyboardInterrupt')
If you want to interrupt the program do :
ctrl+\\
(backslash)
