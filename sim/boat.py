import numpy as np
from typing import Any, Dict, Optional

from tells_environment_dynamics.sim.boat_dynamics import boatDynamics


class Boat:
    def __init__(
            self, 
            name: str,
            dynamics: boatDynamics,
            size: int = 2,
    ) -> None:
        '''
        drone dynamics container class

        input
        -----
        name:str
            name of satellite
        droneDynamics:droneDynamics
            drone dynamics class, required for all drones
        size:float
            size of spacecraft for plotting
        '''

        self.dynamics = dynamics
        self.name = name
        self.size = size

    def reset(
            self,
            init_data: Optional[dict[str,Any]]=None,
    ):
        '''
        reset drone dynamics

        input
        -----
        init_data:Optional[dict[str,Any]]
            dictionary with initial state date to reset dynamics to
        '''

        self.dynamics.reset(init_data)

    def forward_step(self):
        '''
        take forward step for all dynamics objects
        '''

        self.dynamics.forward_step()
    
    def set_ctrl(
            self,
            ctrl: list[float],
    ) -> None:
        '''
        set control in dynamics class

        input
        -----
        ctrl:list[float]
            control input for droneDynamics
        '''

        self.dynamics.set_control(ctrl)


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
            requested attribut from dynamics object (pos,vel,quat,euler,omega,dcm,speed,A,B)
        '''
        if self.dynamics is not None:
            try:
                func_name = 'get_' + attr
                return getattr(self.dynamics, func_name)()
            except:
                try:
                    return getattr(self.dynamics, attr)
                except:
                    print('Error:', attr, 'is not provided in boat dynamics class')

    def delete(self) -> None:

        del self.dynamics
