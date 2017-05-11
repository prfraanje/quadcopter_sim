# quadcopter_sim
simulation of quadcopter using pybullet (calculations) and pyqtgraph (visualisation)

# Dependencies
* [`pybullet`](http://bulletphysics.org/wordpress/)
* [`pyqtgraph`](http://www.pyqtgraph.org/)

# Commment
This code is used for simulating the dynamics of a quadcopter. The simulation
is done in a separate thread, so the parameters and setpoints can be adjusted
interactively. The motors are modelled as force-actuators, no airodynamics
is taken into account. It is used in education, and therefore not complete
on purpose: PID controller needs to be completed and the controller
parameters should be tuned. The simulation is not meant for yaw-control
(i.e. yaw=0)! For yaw-control the inverse kinematics in calculating the
desired roll-pitch-yaw from the desired force-vector [Fx,Fy,Fz] should
be changed.

# Example
Run the file `quadcopter_sim.py` for example in `ipython` and read and change some
parameters:
```
In [1]: %run quadcopter_sim.py
In [2]: qcc.error_pos
In [3]: qcc.ref_pos = np.array([0,0,1])
In [4]: qcc.z_ctrl.Kp
In [5]: qcc.z_ctrl.Kp = 10
```
Tip: look into the comments in `quadcopter_sim.py`
