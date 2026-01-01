import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, Optional
from mpl_toolkits.mplot3d import Axes3D

from tells_environment_dynamics.sim.base_dynamics import baseDynamics

'''
Orbital dynamics for satellite simulation (onlly for circular orbits)
Input arguments are in degrees
'''


class circularOrbit(baseDynamics):
    # Set constant orbital radius for geosynchronous orbit

    def __init__(
          self, 
          semi_major_axis: float, 
          inclination: float = 0, 
          raan: float = 0, 
          arg_periapsis:float = 0, 
          timestep: float = 1.0,
          horizon: int = 10,
          true_anomaly:float = 0
    ) -> None :
        '''
        orbital dynamics class for circular orbit
        propogation is done using simplification only valid
        with eccentricity = 0

        imput
        -----
        semi_major_axis:float
            semi_major_axis (equal to radius for circular orbits) of the orbit in meters
        inclination:float
            inclination of orbit in degrees
        raan:float
            right angle of ascending node (RAAN) of orbit in degrees
        arg_peripasis:float
            argument of periapsis of orbit in degrees
        timestep:float
            size of timestep (seconds)
        horizon:int
            number of timesteps to be take with each forward dyanmics step
        true_anomaly:float
            true anomaly of orbit in degrees
        
        '''
        super().__init__(horizon=horizon,timestep=timestep)

        self.a = semi_major_axis # Orbit semi major axis in meters
        self.eccentricity = 0  # circular orbit
        self.inclination = np.radians(inclination)  # Orbit Inclination in radians
        self.raan = np.radians(raan)  # Right Ascension of Ascending Node in radians
        self.arg_periapsis = np.radians(arg_periapsis)  # Argument of Periapsis in radians
        self.true_anomaly = np.radians(true_anomaly)  # True Anomaly in radians
        self.current_mean_anomaly = np.radians(true_anomaly)
        
        # Constants
        self.mu = 3.986004418e14  # Earth's gravitational parameter (m^3/s^2)
        self.omega = np.sqrt(self.mu / (self.a ** 3))  # Angular velocity (rad/s)
        self.cf: list[list[float]] = np.array([[1.0, 0.0, 0.0],
                                          [0.0, 1.0, 0.0],
                                          [0.0, 0.0, 1.0],
                                          ]) #Global coordinate frame (ECI)
        
        self.time = 0 #initialize time
        self.get_pos()
        self.get_vel()

    def load_new_orbit(
            self,
            semi_major_axis: float, 
            inclination: float = 0, 
            raan: float = 0, 
            arg_periapsis:float = 0, 
        ) -> None:
        '''
        changes orbit parameters
        
        imput
        -----
        semi_major_axis:float
            semi_major_axis (equal to radius for circular orbits) of the orbit in meters
        inclination:float
            inclination of orbit in degrees
        raan:float
            right angle of ascending node (RAAN) of orbit in degrees
        arg_peripasis:float
            argument of periapsis of orbit in degrees
        '''
        self.a = semi_major_axis # Orbit semi major axis in meters
        self.inclination = np.radians(inclination)  # Orbit Inclination in radians
        self.raan = np.radians(raan)  # Right Ascension of Ascending Node in radians
        self.arg_periapsis = np.radians(arg_periapsis)  # Argument of Periapsis in radians

    def reset(
            self
    ) -> None:
        self.current_mean_anomaly = self.true_anomaly
        self.get_pos()
        self.get_vel()
    
    def forward_step(
          self, 
          time: Optional[float] = None,
    ) -> tuple[list[float],list[float]]:
        '''
        input
        -----
        time:float
            time to propagate the orbit forward (seconds), if None will use self.timestep*self.horizon

        output
        ------
        list[float]
            position of orbit in ECEF
        list[float]
            velocity of orbit in ECEF 
        '''
        if time is None:
            self.time += self.timestep*self.horizon
        else:
            self.time += time

        self.current_mean_anomaly = (self.omega * self.time + self.true_anomaly) % (2*np.pi)

        # Print the true anomaly for debugging
        #print(f"Time: {self.time:.2f}s, True Anomaly: {np.degrees(self.true_anomaly):.2f}°")

        self.position_orbit = self.get_pos() #compute position in orbital frame
        self.velocity_orbit = self.get_vel() #compute velocity in orbital frame

        self.position_ecef = self.get_pos_global()
        self.velocity_ecef = self.get_vel_global()

        #Returns the new position and velocity (in ECEF frame) after propogating for time
        return self.position_ecef, self.velocity_ecef

    def transform_orbital_to_ecef(
          self, 
          vector
    ) -> list[float]:
        '''
        Input
        -----
        vector:list[float]
            vector to be transformed to Earth Centered Earth Fixed frame from orbital frame

        Output
        ------
        list[float]
            transformed vector
        '''
        # Define rotation matrices
        R_z_raan = np.array([
            [np.cos(self.raan), -np.sin(self.raan), 0],
            [np.sin(self.raan), np.cos(self.raan), 0],
            [0, 0, 1]
        ])

        R_x_inclination = np.array([
            [1, 0, 0],
            [0, np.cos(self.inclination), -np.sin(self.inclination)],
            [0, np.sin(self.inclination), np.cos(self.inclination)]
        ])

        R_z_arg_periapsis = np.array([
            [np.cos(self.arg_periapsis), -np.sin(self.arg_periapsis), 0],
            [np.sin(self.arg_periapsis), np.cos(self.arg_periapsis), 0],
            [0, 0, 1]
        ])

        # Compute rotation
        #return R_z_raan @ (R_x_inclination @ (R_z_arg_periapsis @ vector))
        return (R_z_raan @ (R_x_inclination @ R_z_arg_periapsis)) @ vector
    
    def transform_ecef_to_orbital(
          self, 
          vector
    ) -> list[float]:
        '''
        Input
        -----
        vector:list[float]
            vector to be transformed to Earth Centered Earth Fixed frame from orbital frame

        Output
        ------
        list[float]
            transformed vector
        '''
        # Define rotation matrices
        R_z_raan = np.array([
            [np.cos(self.raan), -np.sin(self.raan), 0],
            [np.sin(self.raan), np.cos(self.raan), 0],
            [0, 0, 1]
        ])

        R_x_inclination = np.array([
            [1, 0, 0],
            [0, np.cos(self.inclination), -np.sin(self.inclination)],
            [0, np.sin(self.inclination), np.cos(self.inclination)]
        ])

        R_z_arg_periapsis = np.array([
            [np.cos(self.arg_periapsis), -np.sin(self.arg_periapsis), 0],
            [np.sin(self.arg_periapsis), np.cos(self.arg_periapsis), 0],
            [0, 0, 1]
        ])

        # Compute rotation
        #return R_z_raan @ (R_x_inclination @ (R_z_arg_periapsis @ vector))
        return (R_z_raan @ (R_x_inclination @ R_z_arg_periapsis)).T @ vector
    
    def transform_hill_to_orbital(
            self,
            vector,
    ) -> list[float]:
        '''
        Input
        -----
        vector:list[float]
            vector to be transformed to orbital frame from Hill frame

        Output
        ------
        list[float]
            transformed vector
        '''
        dcm = np.array([
            [np.cos(self.current_mean_anomaly), -np.sin(self.current_mean_anomaly), 0],
            [np.sin(self.current_mean_anomaly), np.cos(self.current_mean_anomaly), 0],
            [0,0,1]
        ])

        return dcm @ vector
    
    def transform_orbital_to_hill(
            self,
            vector,
    ) -> list[float]:
        '''
        Input
        -----
        vector:list[float]
            vector to be transformed to orbital frame from Hill frame

        Output
        ------
        list[float]
            transformed vector
        '''
        dcm = np.array([
            [np.cos(self.current_mean_anomaly), -np.sin(self.current_mean_anomaly), 0],
            [np.sin(self.current_mean_anomaly), np.cos(self.current_mean_anomaly), 0],
            [0,0,1]
        ])

        return dcm.T @ vector

    def get_pos(
            self,
    ) -> list[float]:
        '''
        output
        ------
        list[float]
            position of current orbit in orbital frame
        '''
        x_orbit = self.a * np.cos(self.current_mean_anomaly)
        y_orbit = self.a * np.sin(self.current_mean_anomaly)

        self.position_orbit = np.array([x_orbit, y_orbit, 0])

        return self.position_orbit

    def get_vel(
            self, 
    ) -> list[float]:
        '''
        output
        ------
        list[float]
            velocity of current orbit in orbital frame
        '''
        v_orbit = np.sqrt(self.mu / self.a)

        vx = v_orbit * np.sin(self.current_mean_anomaly)
        vy = v_orbit * np.cos(self.current_mean_anomaly)

        # Velocity vector in the orbital frame
        self.velocity_orbit = np.array([-vx, vy, 0])

        return self.velocity_orbit
    
    def get_pos_global(
            self,
    ) -> list[float]:
        '''
        output
        ------
        list[float]
            position in ECEF frame
        '''
        return self.transform_orbital_to_ecef(self.position_orbit)
    
    def get_vel_global(
            self,
    ) -> list[float]:
        '''
        output
        ------
        list[float]
            velocity in ECEF frame
        '''
        return self.transform_orbital_to_ecef(self.velocity_orbit)


