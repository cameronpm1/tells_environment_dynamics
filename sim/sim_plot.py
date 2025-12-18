#Plot simulation results
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from mpl_toolkits.mplot3d.art3d import Line3DCollection


#Clean up code attempt:
class Renderer3D:
     def __init__(self, xlim, ylim, zlim):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_zlim(zlim)
        self.lines_collection = None  # Store the Line3DCollection
        self.points_collection = None  # Store the scatter points

     def clear(self):
        #Clear the current axes while preserving limits.
        self.ax.cla()  # Clear the current axes
        self.ax.set_xlim(self.xlim)
        self.ax.set_ylim(self.ylim)
        self.ax.set_zlim(self.zlim)

     def plot(self,object_data):
         #self.ax.clear() #Clear previous plots
         points = object_data['points']
         lines = object_data['lines']

         for line in lines:
            self.ax.plot([points[line[0]][0],points[line[1]][0]],
                            [points[line[0]][1],points[line[1]][1]],
                            [points[line[0]][2],points[line[1]][2]], color="k")
         
         self.ax.set_xlabel('X')
         self.ax.set_ylabel('Y')
         self.ax.set_zlabel('Z')
         
         plt.draw()
         plt.pause(0.01)

#Clean up code attempt:
class Renderer2D:
     def __init__(self, xlim, ylim):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot()
        self.xlim = xlim
        self.ylim = ylim
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.lines_collection = None  # Store the Line3DCollection
        self.points_collection = None  # Store the scatter points

     def clear(self):
        #Clear the current axes while preserving limits.
        self.ax.cla()  # Clear the current axes
        self.ax.set_xlim(self.xlim)
        self.ax.set_ylim(self.ylim)

     def plot(self,object_data):
         #self.ax.clear() #Clear previous plots
         points = object_data['points']
         lines = object_data['lines']

         for line in lines:
            self.ax.plot([points[line[0]][0],points[line[1]][0]],
                         [points[line[0]][1],points[line[1]][1]], color="k")
         
         self.ax.set_xlabel('X')
         self.ax.set_ylabel('Y')
         
         plt.draw()
         plt.pause(0.01)
         
     
def main():
    renderer = Renderer3D(xlim = [-30,30], ylim = [-30,30], zlim = [-30,30])
    
    #Define a simple line object
    object_data = {
        'points':[[0,0,0],[10,10,10]],
        'lines': [[0,1]]
    }

    renderer.plot(object_data)
    plt.show()


if __name__ == "__main__":
    main()





     