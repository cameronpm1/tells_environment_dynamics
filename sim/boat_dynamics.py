import copy
import scipy
import numpy as np
from scipy.integrate import odeint
from matplotlib import pyplot as plt
from typing import Any, Dict, Optional

from tells_environment_dynamics.sim.base_dynamics import baseDynamics
from tells_environment_dynamics.sim.orbital_dynamics import circularOrbit

class boatDynamics(baseDynamics):

    def __init__(
            self,
            inertial_data: Optional[Dict[str,Any]],
            initial_state_data: Optional[Dict[str,Any]] = None,
            horizon: int = 10,
            timestep: float = 1.0,
    ) -> None:
        '''
        2D dynamics simulation for a boat assuming thrust in forward 
        direction of boat, and torque can be applied

        input
        -----
        timestep:float
            size of timestep (seconds)
        horizon:int
            number of timesteps to be take with each forward dyanmics step
        inertial_data:Dict[str,list[float]]
            inertial parameters/constants of drone
        initial_state_data:Dict[str,list[float]]
            initial translational and rotational parameters of drone
        '''
        super().__init__(horizon=horizon,timestep=timestep)

        if initial_state_data is not None:
            self.pos = initial_state_data['position'] #m
            self.vel = initial_state_data['velocity'] #m/s
            self.hdg = initial_state_data['heading'] #rad/s
            self.omega = initial_state_data['angular_velocity']
        else:
            self.pos = np.array([0.0, 0.0]) #initial position
            self.vel = np.array([0.0, 0.0]) #initial velocity
            self.hdg = np.array([0.0]) #angular velocity vector
            self.omega = np.array([0.0]) #initial quaternion

        #Additional spacecraft data initialization
        if inertial_data is not None:
            self.I11 = float(inertial_data['J_b'])
            self.mass = inertial_data['mass']
            self.l = inertial_data['length']
            self.kf = inertial_data['friction']
        else:
            print('Error: no drone inertial data')
            exit()
        
        #Initialize dynamics parameters
        self.time = 0
        self.state  = None
        self.control = None
        self.state_matrix = None
        self.control_matrix = None
        self.initialize_state()
        self.initialize_control()
        self.initial_state = copy.deepcopy(self.state)
        
    @property
    def A(self):
        '''   
        Linear State Space Representation
        dx/dt = Ax+Bu
        '''
        #Linear state transition matrix
        A = np.zeros((self.state.size,self.state.size))
        #Linear velocty
        A[0,2] = 1
        A[1,3] = 1
        #Angular verlocity
        A[4,5] = 1
        return A

    @property
    def B(self): 
        '''   
        Linear State Space Representation
        dx/dt = Ax+Bu
        '''  
        B = np.zeros((self.state.size,self.control.size))
        #R = self.get_dcm()
        #B[2:4,0:2] = R/self.mass
        B[2,0] = 1/self.mass
        B[3,1] = 1/self.mass
        B[5,2] = 1/self.I11

        return B
    
    @property
    def T(self): 
        '''   
        Linear State Space Representation
        dx/dt = Ax+Bu
        '''  
        T = np.zeros((self.control.size,self.control.size))
        R = self.get_dcm()
        T[0:2,0:2] = R
        T[2,2] = 1.0

        return T

    def initialize_state(self) -> None:
        self.state = np.concatenate((self.pos,self.vel,self.hdg,self.omega), axis=None)
        self.initial_state = self.state

    def initialize_control(self) -> None:
        self.control = np.zeros(3,)
        self.initialcontrol = self.control

    def reset(
            self,
            initial_state_data: Optional[dict[str,Any]]=None,
    ) -> None:
        '''
        inpout
        ------
        initial_state_data:Optional[dict[str,Any]]
            if given, dictionary with pos, vel, omega, and quaternions for setting new initial state
        '''
        self.time = 0
        if initial_state_data is not None:
            self.pos = initial_state_data['position'] #m
            self.vel = initial_state_data['velocity'] #m/s
            self.omega = initial_state_data['heading'] #rad/s
            self.quat = initial_state_data['angular_velocity']
        else:
            self.pos = self.initial_state[0:2]
            self.vel = self.initial_state[2:4]
            self.hdg = self.initial_state[5]
            self.omega = self.initial_state[6]
        self.initialize_state()

    def set_control(
            self,
            control: list[float],
    ) -> None:
        '''
        takes input in boat thrust and change in heading

        input
        -----
        control:list[float]
            control = [T, 0, delta_hdg]
        '''
        self.control[0] = control[0]
        self.state[4] += control[2]
            
    def compute_derivatives(
            self,
            state,
            t,
    ) -> list[float]:
        '''
        compute state derivatives for propogation

        input
        -----
        state:list[float]
            dummy state input (not used, passed by scipy.integrate.odeint)
        time:float
            dummy time variable (not used, passed by scipy.integrate.odeint)

        output
        ------
        list[float]
            dx/dt = [v_x,v_y,a_x,a_y,omega_z,omega'_z]
        '''

        state_matrix = self.A
        control_matrix = self.B
        transform_matrix = self.T

        #velocity cap
        self.state[2:4] = np.clip(self.state[2:4],-15,15)

        # Compute state derivative
        dxdt = np.zeros_like(self.state)
        dxdt = np.matmul(state_matrix, self.state) + np.squeeze(np.matmul(control_matrix, np.matmul(transform_matrix,self.control)))

        #velocity cap
        self.state[2:4] = np.clip(self.state[2:4],-15,15)
        
        return dxdt
    
    def forward_step(self) -> list[float]:
        '''
        propogate dynamics by self.horizon timesteps

        output
        ------
        list[float]
            state = [p_x,p_y,p_z,v_x,v_y,v_z,omega_x,omega_y,omega_z,w,x,y,z]
        '''
        # Define the time range for the integration
        timerange = np.arange(self.time, self.time + (self.timestep * self.horizon), self.timestep)
        
        #apply energy loss due to friction
        self.state[2:4] *= (1-self.kf)
        self.state[5] *= (1-self.kf)

        # Integrate using odeint
        # The compute_derivatives function is passed to odeint
        sol = scipy.integrate.odeint(self.compute_derivatives, self.state, timerange)
        
        # Update the current time and state to the last state in the solution
        self.time += self.timestep * self.horizon
        self.state = sol[-1]  # update state

        return self.state
    
    #Methods to retrieve certain state variables
    def get_pos(self) -> list[float]:
        '''
        output
        ------
        list[float]
            position of drone in cartesian coordinates: pose = [p_x,p_y,p_z]
        '''
        return self.state[0:2]
    
    def get_vel(self) -> list[float]:
        '''
        output
        ------
        list[float]
            velocity of drone in cartesian coordinates: vel = [v_x,v_y,v_z]
        '''
        return self.state[2:4]
    
    def get_speed(self) -> float:
        '''
        output
        ------
        float
            speed of drone: speed = sqrt(v_x^2+v_y^2+v_z^2)
        '''
        return np.linalg.norm(self.state[2:4])
    
    def get_hdg(self) -> float:

        return self.state[4]
    
    def get_omega(self) -> list[float]:
        '''
        output
        ------
        list[float]
            angular velocities of drone: omega = [w_x,w_y,w_z]
        '''
        return self.state[5]
    
    def get_dcm(self) -> list[list[float]]:
        '''
        output
        ------
        list[list[float]]
            2x2 DCM computed from heading (rotation around z-axis)
        '''
        hdg = self.state[4]

        dcm = np.array([
            [np.cos(hdg), -np.sin(hdg)],
            [np.sin(hdg),  np.cos(hdg)]
        ])

        return dcm
    
    def get_state(self) -> list[float]:
        '''
        output
        ------
        list[float]
            full state (6x1 matrix): [x,y,v_x,v_y,theta,theta_dot]
        '''

        return self.state
    
    def get_A(self):
        '''
        output
        ------
        list[list[float]]
            state matrix
        '''
        return self.A
    
    def get_B(self):
        '''
        output
        ------
        list[list[float]]
            control matrix
        '''
        return self.B
