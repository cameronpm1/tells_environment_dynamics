import time
import numpy as np
import matplotlib.pyplot as plt

from tells_environment_dynamics.sim.boat import Boat
from tells_environment_dynamics.sim.drone import Drone
from tells_environment_dynamics.sim.sim_plot import Renderer3D
from tells_environment_dynamics.sim.sim_plot import Renderer2D
from tells_environment_dynamics.sim.satellite import Satellite
from tells_environment_dynamics.sim.cwh_dynamics import CWHDynamics
from tells_environment_dynamics.sim.boat_dynamics import boatDynamics
from tells_environment_dynamics.sim.constellation import Constellation
from tells_environment_dynamics.sim.drone_dynamics import droneDynamics
from tells_environment_dynamics.sim.orbital_dynamics import circularOrbit

box_points = np.array([
    [-0.5, -0.5, -0.5],
    [0.5, -0.5, -0.5],
    [0.5, 0.5, -0.5],
    [-0.5, 0.5, -0.5],
    [-0.5, -0.5, 0.5],
    [0.5, -0.5, 0.5],
    [0.5, 0.5, 0.5],
    [-0.5, 0.5, 0.5]
])
box_lines = np.array([
    (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
    (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
    (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
])

hill_axis_points = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
])
hill_axis_lines = np.array([
    (0,1),(0,2),(0,3)
])

x_points = np.array([
    [0.5, 0.1, 0.0],
    [0.5, -0.1, 0.0],
    [-0.5, 0.1, 0.0],
    [-0.5, -0.1, 0.0],
    [0.1, 0.5, 0.0],
    [-0.1, 0.5, 0.0],
    [0.1, -0.5, 0.0],
    [-0.1, -0.5, 0.0],
])
x_lines = np.array([
    (0,1), (1,3), (3,2), (2,0),
    (4,5), (5,7), (7,6), (6,4),
])

boat_points = np.array([
    [0.0, 0.25],
    [1.5, 0.25],
    [2.0, 0.0],
    [1.5, -0.25],
    [0.0, -0.25]
])

boat_lines = np.array([
    (0,1), (1,2), (2,3), (3,4), (4,0)
])

def make_sat(
        name,
        local=True,
):
    timestep = 0.01
    horizon = 10

    #create orbit dyanmics object 
    orbit = circularOrbit(
        semi_major_axis=46e6,
        raan=45,
        inclination=45,
        arg_periapsis=0,
        timestep=timestep,
        horizon=horizon,
    )

    if local:
        #create local dynamics object 
        initial_state_data = {
            'position': np.array([0.0, 0.0, 0.0]),
            'velocity': np.array([0.0, 0.0, 0.0]),
            'angular_velocity': np.array([0.2, 0.0, 0.0]),
            'quaternion': np.array([1.0, 0.0, 0.0, 0.0]),
        }
        inertial_data = {
            'J_sc': np.array([1.7e4, 2.7e4, 2.7e4]),
            'mass': 4000,
        }
        localDynamics = CWHDynamics(
            orbit=orbit,
            inertial_data=inertial_data,
            initial_state_data=initial_state_data,
            timestep=timestep,
            horizon=horizon,
        )
    else:
        localDynamics = None

    #initialize satellite container class
    sat = Satellite(
        name=name,
        orbitDynamics=orbit,
        localDynamics=localDynamics,
    )

    return sat

def make_drone(
        name,
):
    timestep = 0.001
    horizon = 10

    #create drone dynamics object 
    initial_state_data = {
        'position': np.array([0.0, 0.0, 0.0]),
        'velocity': np.array([0.0, 0.0, 0.0]),
        'angular_velocity': np.array([0.0, 0.0, 0.0]),
        'quaternion': np.array([1.0, 0.0, 0.0, 0.0]),
    }
    inertial_data = {
        'J_qc': np.array([7.5e-3, 7.5e-3, 1.3e-3]),
        'mass': 0.65,
        'arm_length' : 0.23,
        'k_f' : 3.13e-5,
        'k_m' : 7.5e-7,
        'J_r' : 6e-5
    }
    dynamics = droneDynamics(
        inertial_data=inertial_data,
        initial_state_data=initial_state_data,
        timestep=timestep,
        horizon=horizon,
    )

    #initialize satellite container class
    drone = Drone(
        name=name,
        dynamics=dynamics,
    )

    return drone

def make_boat(
        name,
):
    timestep = 0.1
    horizon = 10

    #create drone dynamics object 
    initial_state_data = {
        'position': np.array([0.0, 0.0]),
        'velocity': np.array([0.0, 0.0]),
        'heading': np.array([0.0]),
        'angular_velocity': np.array([0.0]),
    }
    inertial_data = {
        'J_b': 3e4,
        'mass': 4000,
        'length' : 10,
        'friction' : 0.05,
    }
    dynamics = boatDynamics(
        inertial_data=inertial_data,
        initial_state_data=initial_state_data,
        timestep=timestep,
        horizon=horizon,
    )

    #initialize satellite container class
    boat = Boat(
        name=name,
        dynamics=dynamics,
    )

    return boat

def test_local_sat_dynamics():
    '''
    test for localized dynamics, plots rotating cube
    '''
    steps = 50
    sat = make_sat('sat1')

    plot_data = {}
    plot_data['lines'] = box_lines
    plot_data['points'] = box_points
    sat.set_local_ctrl([0,0,0,0.0,0.5,0])

    #start plotter
    renderer = Renderer3D(xlim=[-10, 10], ylim=[-10, 10], zlim=[-10, 10])  
    plt.ion()

    for i in range(steps):
        if i == 20:
            sat.set_local_ctrl([0.0,0,0,0.0,0.0,0])
        #forward step and point computation
        sat.forward_step()
        pos = sat.get_local_attr('pos')
        dcm = sat.get_local_attr('dcm')
        transformed_vertices = np.dot(box_points*sat.size, dcm.T) + pos
        plot_data['points'] = transformed_vertices
        #plotting
        renderer.clear()  # Clear once for all satellites
        renderer.plot(plot_data)  # Plot satellite
        plt.pause(0.05)

    plt.ioff()
    plt.show()

def test_angles():
    '''
    test for localized dynamics, plots rotating cube
    '''
    steps = 500
    sat = make_sat('sat1')

    plot_data = {}
    plot_data['lines'] = box_lines
    plot_data['points'] = box_points
    sat.set_local_ctrl([0,0,0,0.0,0.5,0])

    #start plotter
    renderer = Renderer2D(xlim=[0, steps], ylim=[0, np.pi*2])  
    plt.ion()

    a1 = []
    a2 = []
    a3 = []
    x = []

    for i in range(steps):
        #forward step and point computation
        sat.forward_step()
        angles = sat.get_local_attr('euler')   
        print(angles)
        angles = (angles + np.pi*2) % (np.pi*2)
        print(angles)

        a1.append(angles[0])
        a2.append(angles[1])
        a3.append(angles[2])
        x.append(i)
        plot_data['X'] = x
        plot_data['data'] = [a1]

        #plotting
        renderer.clear()  # Clear once for all satellites
        renderer.plot(plot_data)  # Plot satellite
        plt.pause(0.05)

    plt.ioff()
    plt.show()

def test_orbit_dynamics():
    '''
    test orbital dynamics and dcms, 
    will plot moving orbit with axis for Hill frame
    '''
    steps = 100
    sat = make_sat('sat1',local=False)
    plot_data = {}
    plot_data['lines'] = hill_axis_lines
    plot_data['points'] = hill_axis_points
    #increase sat timestep for plotting
    sat.orbit.timestep = 60
    scale = 10e6 #axis scale

    #start plotter
    renderer = Renderer3D(xlim=[-50e6, 50e6], ylim=[-50e6, 50e6], zlim=[-50e6, 50e6])  
    plt.ion()
    orbit_track = []

    for i in range(steps):
        #forward step and point computation
        sat.forward_step()
        pos = sat.get_global_pos()
        #log orbit pos
        if i == 0:
            orbit_track = np.array([pos])
        else:
            orbit_track = np.append(orbit_track, np.array([pos]), axis=0)
        transformed_vertices = sat.transform_hill_to_ecef(hill_axis_points*scale)
        plot_data['points'] = transformed_vertices + pos
        #plotting
        renderer.clear()  # Clear once for all satellites
        plt.plot(orbit_track[:,0],orbit_track[:,1],orbit_track[:,2]) #plot orbit
        renderer.plot(plot_data)  # Plot axis
        plt.pause(0.05)

    plt.ioff()
    plt.show()

def test_constellation_dynamics():
    '''
    test constellation dynamics
    '''
    steps = 100
    
    constellation = Constellation(
        planes=3,
        num_satellites=27,
        semi_major_axis=2000000,
        timestep=0.5,
    )

    #start plotter
    renderer = Renderer3D(xlim=[-3e6, 3e6], ylim=[-3e6, 3e6], zlim=[-3e6, 3e6])  
    plt.ion()

    for i in range(steps):
        orbit_track = []
        #forward step and point computation
        constellation.forward_step()
        sats = constellation.get_satellites()
        for name,sat in sats:
            orbit_track.append(sat['sat'].get_global_pos())
        #plotting
        renderer.clear()  # Clear once for all satellites
        orbit_track = np.array(orbit_track)
        renderer.ax.scatter(orbit_track[:,0],orbit_track[:,1],orbit_track[:,2],s=10) #plot orbit
        renderer.ax.scatter([0], [0], [0], s=1000, color='green', alpha=0.5) #plot earth
        plt.pause(0.05)
        plt.draw()

    plt.ioff()
    plt.show()

def test_drone_dynamics():
    '''
    test for localized dynamics, plots rotating cube
    '''
    steps = 100
    drone = make_drone('drone1')

    plot_data = {}
    plot_data['lines'] = x_lines
    plot_data['points'] = x_points
    drone.set_ctrl([500.0,502.0,500.0,502.0])

    #start plotter
    renderer = Renderer3D(xlim=[-10, 10], ylim=[-10, 10], zlim=[0, 20])  
    plt.ion()

    for i in range(steps):
        if i == 20:
            drone.set_ctrl([0.0,0,0,0.0,0.0,0])
        #forward step and point computation
        drone.forward_step()
        pos = drone.get_local_attr('pos')
        dcm = drone.get_local_attr('dcm')
        transformed_vertices = np.dot(x_points*drone.size, dcm.T) + pos
        plot_data['points'] = transformed_vertices
        #plotting
        renderer.clear()  # Clear once for all satellites
        renderer.plot(plot_data)  # Plot satellite
        plt.pause(0.05)

    plt.ioff()
    plt.show()

def test_boat_dynamics():
    '''
    test for localized dynamics, plots rotating cube
    '''
    steps = 3000
    boat = make_boat('boat1')

    plot_data = {}
    plot_data['lines'] = boat_lines
    plot_data['points'] = boat_points
    boat.set_ctrl([50.0, 0.0, np.pi/4])

    #start plotter
    renderer = Renderer2D(xlim=[-60, 60], ylim=[-60, 60])  
    plt.ion()

    for i in range(steps):
        if i == 20:
            boat.set_ctrl([0.0,0.0,0.0])
        #forward step and point computation
        boat.forward_step()
        pos = boat.get_local_attr('pos')
        dcm = boat.get_local_attr('dcm')
        transformed_vertices = np.dot(boat_points*boat.size, dcm.T) + pos
        plot_data['points'] = transformed_vertices
        #plotting
        renderer.clear()  # Clear once for all satellites
        renderer.plot(plot_data)  # Plot satellite
        plt.pause(0.05)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    #test_angles()
    #test_local_sat_dynamics()
    #test_orbit_dynamics()
    #test_constellation_dynamics()
    #test_drone_dynamics()
    test_boat_dynamics()


    '''
    testing git submodule imports
    '''