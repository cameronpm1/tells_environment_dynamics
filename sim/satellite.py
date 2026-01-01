import numpy as np
from typing import Any, Dict, Optional

from tells_environment_dynamics.sim.cwh_dynamics import CWHDynamics
from tells_environment_dynamics.sim.orbital_dynamics import circularOrbit


class Satellite:
    def __init__(
            self, 
            name: str,
            orbitDynamics: circularOrbit,
            localDynamics: Optional[CWHDynamics] = None,
            size: float = 2,
    ) -> None:
        '''
        satellite dynamics container class, holds orbital and local dynamics if desired

        input
        -----
        name:str
            name of satellite
        orbitDynamics:circularOrbit
            orbital dynamics class, required for all satellites
        localDynamics:Optional[CWHDynamics]
            local dynamics class, only for simulating dynamics of spacecraft in close proximity to orbit
        size:float
            size of spacecraft for plotting
        '''
        
        #initialize dynamics
        self.orbit = orbitDynamics
        self.localDynamics = localDynamics

        self.name = name
        self.size = size

        if self.localDynamics is not None:
            #timestep_continuity_check 
            if self.orbit.horizon*self.orbit.timestep != self.localDynamics.horizon*self.localDynamics.timestep:
                print('Error:', self.name,  'orbital dynamics and local dynamics have different step size')
                exit()

    def reset(
            self,
            orbit_data: Optional[dict[str,Any]]=None,
            local_data: Optional[dict[str,Any]]=None,
    ):
        if orbit_data is not None:
            self.orbit.load_new_orbit(**orbit_data)
        self.orbit.reset()
        if self.localDynamics is not None:
            self.localDynamics.reset(initial_state_data=local_data)

    def forward_step(self):
        '''
        take forward step for all dynamics objects
        '''
        # Perform a forward step in the satellite's dynamics.
        self.orbit.forward_step()
        if self.localDynamics is not None:
            self.localDynamics.forward_step()

    def get_orbit_pos(self):
        '''
        output
        ------
        list[float]
            satellite orbit position in global ECEF frame
        '''
        return self.orbit.get_pos_global()
        
    def get_global_pos(self):
        '''
        output
        ------
        list[float]
            satellite position in global ECEF frame
        '''
        pos_global = self.orbit.get_pos_global()
        if self.localDynamics is not None:
            pos_local_hill_frame = self.localDynamics.get_pos()
            pos_local_ecef_frame = self.transform_hill_to_ecef([pos_local_hill_frame]).squeeze()
            pos_global += pos_local_ecef_frame
        return pos_global
    
    def get_global_vel(self):
        '''
        output
        ------
        list[float]
            satellite velocity in global ECEF frame
        '''
        vel_global = self.orbit.get_vel_global()
        if self.localDynamics is not None:
            vel_local_hill_frame = self.localDynamics.get_vel()
            vel_local_ecef_frame = self.transform_hill_to_ecef([vel_local_hill_frame]).squeeze()
            vel_global += vel_local_ecef_frame
        return vel_global
    
    def transform_hill_to_ecef(
            self,
            vectors: list[list[float]],
    ):
        '''
        input
        -----
        vector:list[list[float]]
            list of vectors in Hill frame to be transformed to ECEF

        output
        ------
        list[float]
            transformed vector in ECEF frame
        '''
        transformed_vectors = []
        for i,vector in enumerate(vectors):
            vec_orbit_frame = self.orbit.transform_hill_to_orbital(vector)
            vec_ecef_frame = self.orbit.transform_orbital_to_ecef(vec_orbit_frame)
            transformed_vectors.append(vec_ecef_frame)
        return np.array(transformed_vectors)
    
    def transform_ecef_to_hill(
            self,
            vectors: list[list[float]],
    ):
        '''
        transforms ECEF frame to hill, ONLY a rotation matrix (no translation
        assumes coordiante frames are ontop of each other)

        input
        -----
        vector:list[list[float]]
            list of vectors in Hill frame to be transformed to ECEF

        output
        ------
        list[float]
            transformed vector in ECEF frame
        '''
        transformed_vectors = []
        for i,vector in enumerate(vectors):
            vec_orbit_frame = self.orbit.transform_ecef_to_orbital(vector)
            vec_hill_frame = self.orbit.transform_orbital_to_hill(vec_orbit_frame)
            transformed_vectors.append(vec_hill_frame)
        return np.array(transformed_vectors)
    
    def set_local_ctrl(
            self,
            ctrl: list[float],
    ) -> None:
        '''
        input
        -----
        ctrl:list[float]
            control input for localDynamics
        '''
        if self.localDynamics is not None:
            self.localDynamics.set_control(ctrl)
        else:
            print('Error:', self.name, 'does not have local dynamics')


    def get_local_attr(
            self,
            attr: str,
    ):
        '''
        input
        -----
        attr:str
            localDynamics attribut to be collected (pos,vel,speed,omega,quat,dcm)

        output
        ------
        list[float]
            requested attribut from localDynamics object
        '''
        if self.localDynamics is not None:
            func_name = 'get_' + attr
            return getattr(self.localDynamics, func_name)()
        else:
            print('Error:', self.name, 'does not have local dynamics')

    def delete(self) -> None:
        del self.orbit
        if self.localDynamics is not None:
            del self.localDynamics

    


    
    