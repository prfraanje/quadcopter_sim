# Simulation of a quadcopter autopilot

# Author: Rufus Fraanje, p.r.fraanje@hhs.nl
# Date:   17/04/2017

# This code is to serve the UAVREG tutorial in the 
# 2nd year Mechatronics program of the Hague University 
# of Applied Sciences.
# The code can be used for free, but comes with no warrantees.
# The author will not take any responsibility for the correctness
# of completeness of this code.
# In fact parts of the code are deleted for the purpose of the
# UAVREG tutrial, and meant to be finished by students.

# import pyqtgraph for the fast 3D graphics using these lines:
# please first install pyqtgraph and OpenGL using
# pip install OpenGL OpenGL_accelerate pyqtgraph
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl
# import numpy for vector calculations
import numpy as np
# import trigonometric functions from math
# for single-value evaluation these are faster than
# the ones from numpy
from math import atan2, acos, asin
# import pybullet for physics simulation
import pybullet as p

# import Thread from threading to create threads
# threads are parts of a program than run parallel to eachother
# and are allowed to share data
# using threads we can run the simulation in a separate thread
# and use the python command line to interact with the simulation
# in this way we can change references (setpoints) and controller
# parameters while the simulation is still running and we see the
# effect directly
from threading import Thread

# import the time module, mainly to use the sleep function
# and some time evaluation functions
import time

# load some utility functions
# bullet2pyqtgraph is a function to convert objects from bullet to 
# pyqtgraph, when using this function for meshes of 3D CAD models 
# make sure you have the python module trimesh as well
# install it using the command: pip install trimesh
# quaternion2axis_angle and quaternion2rotation_matrix are two
# functions to convert a quaternion to an axis-angle and a rotation 
# matrix, which are all different ways to describe an orientation.
from util import bullet2pyqtgraph, quaternion2axis_angle, quaternion2rotation_matrix

######################################################################################
# definition of PID controller class
######################################################################################
class PIDcontroller:
    #construction of the object PIDcontroller:
    def __init__(self,Kp,Ki,Kd,dT,e_prev,e_int):
        self.Kp = Kp         # proportional gain
        self.Ki = Ki         # integral gain
        self.Kd = Kd         # derivative gain

        self.dT = dT         # sampling time
        self.inv_dT = 1/dT   # for speeding up calculations (multiplying is
                             # faster than dividing)
        self.e_prev = e_prev # previous error signal
        self.e_int  = e_int  # integral of error up to now

    # calculation of the PID control signal u and update of the items in memory
    # CHANGE THIS: the current definition of the function calc_control
    # is not correct, and should be changed to calculate a PID control action
    def calc_control(self,e_now):
        e_int = 0
        de    = 0

        # PID control signal: Kp * e_now + Ki * e_integral + Kd * e_derivative
        u     = 0

        # update memory (state) of PIDcontroller:
        self.e_prev = e_now   # next time, now is previous
        self.e_int  = e_int   # keep the integration of the error up to now
        return u

######################################################################################
# definition of quadcopter control (autopilot) class which makes use of
# PIDcontroller class
######################################################################################
class quadcopter_control:
    # function used to contruct quadcopter_control objects, PID controller
    # parameters Kp, Ki and Kd are 6-dimensional vectors for 
    # x-, y-, z-, pitch-, roll-, and yaw-controllers.
    def __init__(self,Kp,Ki,Kd):
        global Tsample_control
        # controller along global x-, y- and z-axis
        self.x_ctrl = PIDcontroller(Kp[0],Ki[0],Kd[0],Tsample_control,0,0)
        self.y_ctrl = PIDcontroller(Kp[1],Ki[1],Kd[1],Tsample_control,0,0)
        self.z_ctrl = PIDcontroller(Kp[2],Ki[2],Kd[2],Tsample_control,0,0)
        # roll is rotation about quadcopters x-axis
        # pitch is rotation about quadcopters y-axis
        # yaw is rotation about quadcopters z-axis
        self.roll_ctrl  = PIDcontroller(Kp[3],Ki[3],Kd[3],Tsample_control,0,0)
        self.pitch_ctrl = PIDcontroller(Kp[4],Ki[4],Kd[4],Tsample_control,0,0)
        self.yaw_ctrl   = PIDcontroller(Kp[5],Ki[5],Kd[5],Tsample_control,0,0)

        # position (x, y, z), and rpy (roll, pitch, yaw) reference:
        self.ref_pos = np.zeros(3)
        self.ref_rpy = np.zeros(3)
        # offset op roll,pitch,yaw referentie ten behoeve van tunen:
        self.ref_rpy_offset = np.zeros(3)

        self.error_pos = np.zeros(3)
        self.error_rpy = np.zeros(3)

        # krachten voor de 4 actuatoren
        self.force_act1 = 0.
        self.force_act2 = 0.
        self.force_act3 = 0.
        self.force_act4 = 0.
        # moment ten behoeve van yaw-control:
        self.moment_yaw = 0.

        # if true continue simulations
        self.sim    = True
        self.sample = 0
        self.hold   = True

    ##################################################################################
    # main function for quadcopter autopilot
    # inputs are pos_meas (measured position) and quaternion_meas (measured
    # orientation), the function returns the forces for the 4 actuators and
    # the moment for the yaw-control
    # In this function you can include gravity compensation, but no need to
    # change the function in other ways
    ##################################################################################
    def update_control(self,pos_meas,quaternion_meas):
        global quadcopterId, gravity, mass
        R_meas   = quaternion2rotation_matrix(quaternion_meas)
        rpy_meas = p.getEulerFromQuaternion(quaternion_meas)

        self.error_pos  = self.ref_pos - pos_meas

        # calc. desired force in global x,y,z coordinates:
        force_pos = np.zeros(3)
        force_pos[0] = self.x_ctrl.calc_control(self.error_pos[0])
        force_pos[1] = self.y_ctrl.calc_control(self.error_pos[1])
        force_pos[2] = self.z_ctrl.calc_control(self.error_pos[2])
        # gravity compensation
        force_pos[2] += gravity*mass

        #############################
        # math1 block               #
        #############################
        sign_z =  np.sign(force_pos[2])

        # transform force to quadcopters coordinate frame
        force_pos_local = np.dot(R_meas.T,force_pos.reshape(3,1)).reshape(3)
        #  force_pos_local[2] = thrust, so for each actuator one quarter
        quarter_thrust = 0.25*force_pos_local[2]

        if sign_z == 0: sign_z = 1
        norm_F = np.linalg.norm(force_pos)
        # following equations only hold for yaw = 0!
        self.ref_rpy[0] = asin(-sign_z*force_pos[1]/norm_F)
        self.ref_rpy[1] = atan2(sign_z*force_pos[0],sign_z*force_pos[2])
        # setpoint for yaw = 0:
        self.ref_rpy[2] = 0.
        # to enhance robustness, do not let absolute reference angle be greater
        # than 30 degrees (pi/6 radians)
        if self.ref_rpy[0] < -np.pi/6: self.ref_rpy[0] = -np.pi/6
        if self.ref_rpy[1] < -np.pi/6: self.ref_rpy[1] = -np.pi/6
        if self.ref_rpy[0] > np.pi/6: self.ref_rpy[0] = np.pi/6
        if self.ref_rpy[1] > np.pi/6: self.ref_rpy[1] = np.pi/6

        #############################
        # end of math1 block        #
        #############################

        self.error_rpy = self.ref_rpy_offset + self.ref_rpy - rpy_meas

        moments = np.zeros(3)
        # roll pitch yaw control:
        moments[0] = self.roll_ctrl.calc_control(self.error_rpy[0])
        moments[1] = self.pitch_ctrl.calc_control(self.error_rpy[1])
        # though we will not change the yaw, we need a yaw-controller to stabilize
        # the yaw at zero radians
        moments[2] = self.yaw_ctrl.calc_control(self.error_rpy[2])


        #############################
        # math2 block               #
        #############################
        factor         = .5/arm_length
        self.force_act1 = quarter_thrust - factor*moments[1]
        self.force_act3 = quarter_thrust + factor*moments[1]
        self.force_act2 = quarter_thrust + factor*moments[0]
        self.force_act4 = quarter_thrust - factor*moments[0]
        self.moment_yaw = moments[2]

        # apparantly the forces for applyExternalForce are not in LINK_FRAME, but the positions are
        # so transform forces in right direction (in world-space coordinates)
        force_act1 = self.force_act1*R_meas[:,2]
        force_act2 = self.force_act2*R_meas[:,2]
        force_act3 = self.force_act3*R_meas[:,2]
        force_act4 = self.force_act4*R_meas[:,2]

        #############################
        # end of math2 block        #
        #############################

        return force_act1,force_act2,force_act3,force_act4,moments[2]


# this function is repeatedly evaluated in a separate thread
# and evualates the control-law and updates the physics
# (no need to change this)
def update_physics(delay,quadcopterId,quadcopter_controller):
    while quadcopter_controller.sim:
        start = time.perf_counter();

        # the controllers are evaluated at a slower rate, only once in the 
        # control_subsample times the controller is evaluated
        if quadcopter_controller.sample == 0:
            quadcopter_controller.sample = control_subsample
            pos_meas,quaternion_meas = p.getBasePositionAndOrientation(quadcopterId)
            force_act1,force_act2,force_act3,force_act4,moment_yaw = quadcopter_controller.update_control(pos_meas,quaternion_meas)
        else:
            quadcopter_controller.sample -= 1

        # apply forces/moments from controls etc:
        # (do this each time, because forces and moments are reset to zero after a stepSimulation())
        p.applyExternalForce(quadcopterId,-1,force_act1,[arm_length,0.,0.], p.LINK_FRAME)
        p.applyExternalForce(quadcopterId,-1,force_act2,[0.,arm_length,0.], p.LINK_FRAME)
        p.applyExternalForce(quadcopterId,-1,force_act3,[-arm_length,0.,0.],p.LINK_FRAME)
        p.applyExternalForce(quadcopterId,-1,force_act4,[0.,-arm_length,0.],p.LINK_FRAME)

        # for the yaw-control:
        p.applyExternalTorque(quadcopterId,-1,[0.,0.,moment_yaw],p.LINK_FRAME)

        # do simulation with pybullet:
        p.stepSimulation()

        # delay than repeat
        calc_time = time.perf_counter()-start
        if ( calc_time > 1.2*delay ):
            #print("Time to update physics is {} and is more than 20% of the desired update time ({}).".format(calc_time,delay))
            pass
        else:
            # print("calc_time = ",calc_time)
            while (time.perf_counter()-start < delay):
                time.sleep(delay/10)

# this function is to update the window (no need to change this)
def update_window():
    global quadcopterId, quadcopterMesh, w
    pos_meas,quat_meas = p.getBasePositionAndOrientation(quadcopterId)
    angle,x,y,z = quaternion2axis_angle(quat_meas)
    quadcopterMesh.setTransform(np.eye(4).flatten())
    quadcopterMesh.rotate(np.degrees(angle),x,y,z,local=True)
    quadcopterMesh.translate(pos_meas[0],pos_meas[1],pos_meas[2])
    window.update()

################################################################################
# The above is just definition of classes and functions
# Here actual python script for the simulation starts
################################################################################

# Definition of update times (in sec.) for quadcopter physics, controller and 
# window refreshing
Tsample_physics    = 0.0001
control_subsample  = 50
Tsample_control    = control_subsample * Tsample_physics
Tsample_window     = 0.02

# definition of number of constants: 
gravity = 9.8
# arm length, mass and moments of inertia of quadcopter
arm_length       = 0.1
mass             = 0.5
Ixx = Iyy        = 0.0023
Izz              = 0.004

# creation of pyqtgraph 3D graphics window
# with a ground plane and coordinate frame (global axis)
window = gl.GLViewWidget()
window.show()
window.setWindowTitle('Bullet Physics example')
grid = gl.GLGridItem()
window.addItem(grid)
global_axis = gl.GLAxisItem()
global_axis.updateGLOptions({'glLineWidth':(4,)})
window.addItem(global_axis)
window.update()

# configure pybullet and load plane.urdf and quadcopter.urdf
physicsClient = p.connect(p.DIRECT)  # pybullet only for computations no visualisation
p.setGravity(0,0,-gravity)
p.setTimeStep(Tsample_physics)
# disable real-time simulation, we want to step through the
# physics ourselves with p.stepSimulation()
p.setRealTimeSimulation(0)
planeId = p.loadURDF("plane.urdf",[0,0,0],p.getQuaternionFromEuler([0,0,0]))
quadcopterId = p.loadURDF("quadrotor.urdf",[0,0,1],p.getQuaternionFromEuler([0,0,0]))

# do a few steps to start simulation and let the quadcopter land safely
for i in range(int(2/Tsample_physics)):
    p.stepSimulation()

# create a pyqtgraph mesh from the quadcopter to visualize
# the quadcopter in the 3D pyqtgraph window
quadcopterMesh = bullet2pyqtgraph(quadcopterId)[0]
window.addItem(quadcopterMesh)
window.update()

# Initialize PID controller gains:
Kp = np.zeros(6)
Ki = np.zeros(6)
Kd = np.zeros(6)

# give them values:
# x-y-z controlers:
Kp[0] = 0
Kp[1] = 0
Kp[2] = 0

Kd[0] = 0
Kd[1] = 0
Kd[2] = 0

Ki[0] = 0
Ki[1] = 0
Ki[2] = 0

# roll-pitch-yaw controlers (yaw is already prefilled):
Kp[3] = 0
Kp[4] = 0
Kp[5] = 25.6

Kd[3] = 0
Kd[4] = 0
Kd[5] = 1.28

Ki[3] = 0
Ki[4] = 0
Ki[5] = 0

# create the quadcopter control (i.e. the autopilot) object
# which is named qcc
qcc = quadcopter_control(Kp,Ki,Kd)

# start the update_physics that updates both physics and control
# in a separate thread
# this allows us to let the physics+control simulation run 
# completely in the background and we keep a python command line
# which can be used to interact with the quadcopter
# e.g. we can manually change setpoints and change control parameters
thread_physics = Thread(target=update_physics,args=(Tsample_physics,quadcopterId,qcc))
# start the thread:
thread_physics.start()

# the graphics window is updated every Tsample_window seconds
# using a timer function from the Qt GUI part of pyqtgraph
# this also runs in the background, but at a much lower speed than
# the physics and control updates.
timer_window = QtCore.QTimer()
timer_window.timeout.connect(update_window)
timer_window.start(int(1/Tsample_window))

# END
# Tips for using:
