import numpy as np
from typing import Any, Dict, Optional

from tells_environment_dynamics.sim.satellite import Satellite
from tells_environment_dynamics.sim.orbital_dynamics import circularOrbit

'''
Creates a satellite constellation based on the Phase 1 Starlink constellation, 
whos parameters are given by "Laser Inter-Satellite Links in a Starlink Constellation", 
by A. Chaudry & H. Yanikomeroglu, Feb. 2021
https://www.researchgate.net/publication/349641367_Laser_Inter-Satellite_Links_in_a_Starlink_Constellation
'''

class Constellation():
    def __init__(
            self, 
            planes: int,
            num_satellites: int,
            semi_major_axis: float,
            start_anomaly: Optional[float] = None,
            inclination: float = 90,
            timestep: float = 1.0,
            horizon: int = 10,
            stagger: bool = True,
    ) -> None:
        '''
        satellite constellation class, responsible for initializing and holding 
        satellite objects

        input
        -----
        planes:int
            number of orbital planes in constellation
        num_satellites:int
            number of satellites in constellation, must be product of planes
        semi_major_axis:float
            semi major axis (equal to radius) of orbits in constellation 
        inclination:float
            inclination of orbits in constellation (in degrees)
        timestep:float
            timestep for orbit dyanmics class
        horizon:int
            number of timesteps to take in each forward step
        '''
        #if num_satellites%planes != 0:
        #    print('Error: num_satellites ('+str(num_satellites)+') must be divisible by planes ('+str(planes)+')')
        #    exit()

        #initialize local variables
        self.planes = planes
        self.num_satellites = num_satellites
        self.inclination = inclination
        self.a = semi_major_axis
        self.timestep = timestep
        self.horizon = horizon
        self.stagger = stagger

        if start_anomaly is None:
            self.start_anomaly = 0
        else: 
            self.start_anomaly = start_anomaly

        self.initialize_constellation()

    def initialize_constellation(self) -> None:
        '''
        initializes all satellite objects in constellation
        '''
        self.satellites = {}
        raan_range = np.linspace(0,180,self.planes + 1) 
        arg_periapsis_range1 = np.linspace(0,360,int(self.num_satellites/self.planes) + 1) 
        arg_periapsis_range2 = np.linspace(0,360,int((self.num_satellites+1)/self.planes) + 1) 
        self.anomaly_gap = arg_periapsis_range1[0] - arg_periapsis_range1[1]


        label = 0 #sat label counter

        for i,raan in enumerate(raan_range[1:]):
            if i%2 == 0 and self.stagger:
                periapsis_start = self.anomaly_gap/2 
            else:
                periapsis_start = 0
            if i < self.num_satellites%self.planes:
                arg_periapsis_range = arg_periapsis_range2
            else:
                arg_periapsis_range = arg_periapsis_range1
            for arg_periapsis in arg_periapsis_range[1:]:
                name = 'sat'+str(label)
                #initialize orbital dynamics
                orbit = circularOrbit(
                    semi_major_axis=self.a,
                    inclination=self.inclination,
                    raan=raan,
                    arg_periapsis=arg_periapsis + periapsis_start + self.start_anomaly,
                    timestep=self.timestep,
                    horizon=self.horizon,
                )
                #initialize satellite container class
                sat = Satellite(
                    name=name,
                    orbitDynamics=orbit,
                )
                #dict container for specific satellite data
                satellite_dict = {
                    'sat' : sat,
                    'raan' : raan,
                    'arg_periapsis' : arg_periapsis,
                }
                self.satellites[name] = satellite_dict

                label += 1

    def forward_step(self) -> None:
        '''
        propogate all satellite orbits
        '''
        for name,sat in self.satellites.items():
            sat['sat'].forward_step()

    def reset(self) -> None:
        '''
        reset all satellites
        '''
        for name,sat in self.satellites.items():
            sat['sat'].reset()

    def get_satellites(
            self,
    ) -> Dict[str,Any]:
        '''
        output
        ------
        Dict[str,Any]
            satellite dictionary
        '''
        return self.satellites.items()
    
    def get_satellite(
            self,
            name: str,
    ) -> Satellite:
        '''
        input
        -----
        name:str
            name of satellite

        output
        ------
        Satellite
        '''
        return self.satellites.get(name)
    
    def delet(self) -> None:
        '''
        delete all satellite classes
        '''
        for name,sat in self.satellites.items():
            sat['sat'].delete()
            del sat['sat']

    


'''
def visualize_constellation(orbits, time_steps):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Loop through each orbit and propagate positions for visualization
    for orbit in orbits:
        positions = []
        for t in time_steps:
            position, _ = orbit.forward_step()
            positions.append(position)

        positions = np.array(positions)
        
        # Plotting the orbit
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label=f'Orbit (Inclination: {np.degrees(orbit.inclination):.1f}°, RAAN: {np.degrees(orbit.raan):.1f}°)', alpha=0.5)

    # Earth representation (simplified)
    earth_radius = 6371e3  # Earth's radius in meters
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x_earth = earth_radius * np.outer(np.cos(u), np.sin(v))
    y_earth = earth_radius * np.outer(np.sin(u), np.sin(v))
    z_earth = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))
    
    ax.plot_wireframe(x_earth, y_earth, z_earth, color='green', alpha=0.5, label='Earth Surface')

    # Labels and settings
    ax.set_xlim([-7e6, 7e6])
    ax.set_ylim([-7e6, 7e6])
    ax.set_zlim([-7e6, 7e6])
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Starlink Constellation Orbits')
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio

    plt.show()

if __name__ == "__main__":
    num_planes = 24
    num_sats_per_plane = 66
    inclination = 53  # in degrees
    semi_major_axis = 550e3 + 6371e3  # altitude in meters + Earth's radius
    plane_spacing = 360 / num_planes  # spacing between planes in degrees
    sat_spacing = 360 / num_sats_per_plane  # spacing between satellites in degrees
    
    # Create the constellation
    constellation = create_starlink_constellation(num_planes, num_sats_per_plane, inclination, semi_major_axis, plane_spacing, sat_spacing)
    
    # Define time steps for visualization (for one complete orbit)
    time_steps = np.linspace(0, 86400, 360)  # 86400 seconds = 1 day, 360 points

    # Visualize the constellation
    visualize_constellation(constellation, time_steps)
'''