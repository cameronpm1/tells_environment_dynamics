import numpy as np

class baseDynamics:
    # Set constant orbital radius for geosynchronous orbit

    def __init__(
          self, 
          horizon: int,
          timestep: float
    ) -> None :
        
        '''
        base dynamics class

        imput
        -----
        timestep:float
            size of timestep (seconds)
        horizon:int
            number of timesteps to be take with each forward dyanmics step
        
        '''

        self.horizon = horizon
        self.timestep = timestep

    def reset(self):
        pass

    def forward_step(self):
        pass

    def get_pos(self):
        pass

    def get_vel(self):
        pass