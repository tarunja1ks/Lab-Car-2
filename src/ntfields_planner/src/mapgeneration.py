import matplotlib
import numpy as np
from pyoctomap import OcTree
from scipy.spatial import cKDTree
import mlx.core as mlx


def indoor_map_gen():
    
    return
def outdoor_map_gen():
    return 

resolution=0.05
tree=OcTree(0.05)
tree.readBinary("/Users/tarunjaikumar/Documents/Robotics-UCSD/ERL-Racecar/catkin_ws/src/mushr_gazebo/exports/octree.bt")

res = tree.getResolution()
print("loaded octree: resolution =", res)
all_points=[]
for leaf in tree.begin_leafs():
    coord = leaf.getCoordinate()
    if(tree.isNodeOccupied(leaf)):
        center=leaf.getCoordinate()
        n=int(leaf.getSize()/resolution)+1
        x=np.linspace(center[0]-leaf.getSize()/2,center[0]+leaf.getSize()/2,n)
        y=np.linspace(center[1]-leaf.getSize()/2,center[1]+leaf.getSize()/2,n)
        X, Y=np.meshgrid(x, y)
        leaf_points=np.stack([X.reshape(-1), Y.reshape(-1)], axis=1)
        all_points.append(leaf_points)
        
        
points = np.concatenate(all_points, axis=0)

points=cKDTree(points)

print(points)
    



