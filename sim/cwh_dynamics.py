import copy
import scipy
import numpy as np
from scipy.integrate import odeint
from matplotlib import pyplot as plt
from typing import Any, Dict, Optional

from tells_environment_dynamics.sim.base_dynamics import baseDynamics
from tells_environment_dynamics.sim.orbital_dynamics import circularOrbit

class CWHDynamics(baseDynamics):

    def __init__(
            self,
            orbit: circularOrbit,
            inertial_data: Optional[Dict[str,Any]],
            timestep: float = 1.0,
            horizon: int = 10,
            initial_state_data: Optional[Dict[str,Any]] = None,
    ) -> None:
        '''
        dynamics simulation for a satellite in orbit
        dynamics equations come from:
        A System to Evade Uncooperative Satellites using
        Radio Frequency Informed Reinforcement Learning
        Policies (Mehlman & Falco, 2024)
        units in m, s, kg

        input
        -----
        timestep:float
            size of timestep (seconds)
        horizon:int
            number of timesteps to be take with each forward dyanmics step
        initial_state_data:Dict[str,list[float]]
            initial translational and rotational parameters of spacecraft
        spacecraft_data:Dict[str,float]
            initial inertial spacecraft data (moments of inertia and mass)
        '''
        super().__init__(horizon=horizon,timestep=timestep)

        if initial_state_data is not None:
            self.pos = initial_state_data['position'] #m
            self.vel = initial_state_data['velocity'] #m/s
            self.omega = initial_state_data['angular_velocity'] #rad/s
            self.quat = initial_state_data['quaternion']
        else:
            self.pos = np.array([0.0, 0.0, 0.0]) #initial position
            self.vel = np.array([0.0, 0.0, 0.0]) #initial velocity
            self.omega = np.array([0.0, 0.0, 0.0]) #angular velocity vector
            self.quat = np.array([1.0, 0.0, 0.0, 0.0]) #initial quaternion

        #Additional spacecraft data initialization
        if inertial_data is not None:
            self.I11 = inertial_data['J_sc'][0]
            self.I22 = inertial_data['J_sc'][1]
            self.I33 = inertial_data['J_sc'][2]
            self.mass = inertial_data['mass']
        else:
            print('Error: no spacecraft inertial data')
            exit()

        #Initialize orbit parameters/constants
        self.G = 6.67430e-11  
        self.m_earth = 5.972e24  # Mass of Earth (kg)
        self.mu = 3.986004418e14
        self.a = orbit.a #Geostationary orbit radius, m
        self.n = np.sqrt(self.mu/(self.a**3)) #Spacecraft orbital rate
        
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
        A[0,3] = 1
        A[1,4] = 1
        A[2,5] = 1
        #Linear acceleration
        A[3,0] = 3*self.n**2
        A[3,4] = 2*self.n
        A[4,3] = -2*self.n
        A[5,2] = -self.n**2
        #Angular acceleration
        omega1 = self.state[6]
        omega2 = self.state[7]
        omega3 = self.state[8]
        A[6,7] = -(self.I33-self.I22)/self.I11*omega3
        A[7,6] = -(self.I11-self.I33)/self.I22*omega3
        A[8,6] = -(self.I22-self.I11)/self.I33*omega2
        #Quaternion Derivatives
        #q_dot1
        A[9,11] = 1/2*omega2 
        A[9,12] = 1/2*omega1 
        A[9,10] = -1/2*omega3
        #q_dot2
        A[10,9] = 1/2*omega3
        A[10,12] = 1/2*omega2
        A[10,11] = -1/2*omega1
        #q_dot3
        A[11,10] = 1/2*omega1
        A[11,12] = 1/2*omega3
        A[11,9] = -1/2*omega2
        #q_dot4
        A[12,9] = -1/2*omega1
        A[12,10] = -1/2*omega2
        A[12,11] = -1/2*omega3
        return A

    @property
    def B(self): 
        '''   
        Linear State Space Representation
        dx/dt = Ax+Bu
        '''  
        B = np.zeros((self.state.size,self.control.size))
        B[3,0] = 1/self.mass
        B[4,1] = 1/self.mass
        B[5,2] = 1/self.mass
        B[6,3] = 1/self.I11
        B[7,4] = 1/self.I22
        B[8,5] = 1/self.I33
        return B

    def initialize_state(self) -> None:
        self.state = np.concatenate((self.pos,self.vel,self.omega,self.quat), axis=None)
        self.initial_state = self.state

    def initialize_control(self) -> None:
        self.control = np.zeros(6,)
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
            self.omega = initial_state_data['angular_velocity'] #rad/s
            self.quat = initial_state_data['quaternion']
        else:
            self.pos = self.initial_state[0:3]
            self.vel = self.initial_state[3:6]
            self.omega = self.initial_state[6:9]
            self.quat = self.initial_state[9:13]
        self.initialize_state()

    def set_control(
            self,
            control: list[float],
    ) -> None:
        '''
        input
        -----
        control:list[float]
            control = [T_x,T_y,T_z,M_x,M_y,M_z]
        '''
        for i,c in enumerate(control):
            self.control[i] = c

            
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
            dx/dt = [v_x,v_y,v_z,a_x,a_y,a_z,omega'_x,omega'_y,omega'_z,w',x',y',z']
        '''
        pos = self.state[0:3]
        vel = self.state[3:6]
        omega = self.state[6:9]
        quat = self.state[9:13]

        # Compute quaternion derivative
        q_dot = 0.5 * np.array([
            -quat[1] * omega[0] - quat[2] * omega[1] - quat[3] * omega[2],
            quat[0] * omega[0] + quat[2] * omega[2] - quat[3] * omega[1],
            quat[0] * omega[1] - quat[1] * omega[2] + quat[3] * omega[0],
            quat[0] * omega[2] + quat[1] * omega[1] - quat[2] * omega[0]
        ])

        state_matrix = self.A
        control_matrix = self.B

        # Compute state derivative
        dxdt = np.zeros_like(self.state)
        dxdt[0:3] = vel
        dxdt[3:6] = np.matmul(state_matrix[3:6, :], self.state) + np.squeeze(np.matmul(control_matrix[3:6, :], self.control))
        dxdt[6:9] = np.matmul(state_matrix[6:9, :], self.state) + np.squeeze(np.matmul(control_matrix[6:9, :], self.control))
        dxdt[9:13] = q_dot  # Update quaternion

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
        
        # Integrate using odeint
        # The compute_derivatives function is passed to odeint
        sol = scipy.integrate.odeint(self.compute_derivatives, self.state, timerange)
        
        # Update the current time and state to the last state in the solution
        self.time += self.timestep * self.horizon
        self.state = sol[-1]  # update state
        
        return sol
    
    #Methods to retrieve certain state variables
    def get_pos(self) -> list[float]:
        '''
        output
        ------
        list[float]
            position of drone in cartesian coordinates: pose = [p_x,p_y,p_z]
        '''
        return self.state[0:3]
    
    def get_vel(self) -> list[float]:
        '''
        output
        ------
        list[float]
            velocity of drone in cartesian coordinates: vel = [v_x,v_y,v_z]
        '''
        return self.state[3:6]
    
    def get_speed(self) -> float:
        '''
        output
        ------
        float
            speed of drone: speed = sqrt(v_x^2+v_y^2+v_z^2)
        '''
        return np.linalg.norm(self.state[3:6])
    
    def get_omega(self) -> list[float]:
        '''
        output
        ------
        list[float]
            angular velocities of drone: omega = [w_x,w_y,w_z]
        '''
        return self.state[6:9]
    
    def get_quat(self) -> list[float]:
        '''
        output
        ------
        list[float]
            quaternions of drone: quat = [w,x,y,z]
        '''
        return self.state[9:13]
    
    def get_dcm(self) -> list[list[float]]:
        '''
        output
        ------
        list[list[float]]
            3x3 3-2-1 DCM computed from quaternions
        '''
        quat = self.get_quat()
        norm = np.linalg.norm(quat)
        w, x, y, z = quat/norm #self.get_quat()
        # Create the DCM

        dcm = np.array([
            [w**2+x**2-y**2-z**2, 2*(x*y - z*w), 2*(x*z + w*y)],
            [2*(x*y + w*z), w**2-x**2+y**2-z**2, 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), w**2-x**2-y**2+z**2]
        ])
        return dcm
    
    def get_euler(self) -> list[float]:
        '''
        output
        ------
        list[float]
            euler angles computed from quaternions
        '''

        w, x, y, z = self.get_quat()

        phi = np.arctan2(2*(w*x + y*z),1-2*(x**2+y**2))
        theta = -np.pi/2 + 2*np.arctan2(1 + 2*(w*y - x*z),1 - 2*(w*y - x*z))
        psi = np.arctan2(2*(w*z + x*y),1-2*(y**2+z**2))

        return np.array([phi, theta, psi])
    
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
    
    def get_n(self) -> float:
        '''
        output
        ------
        float:
            n of orbit: sqrt(mu/a^3)
        '''

        return self.n

    
