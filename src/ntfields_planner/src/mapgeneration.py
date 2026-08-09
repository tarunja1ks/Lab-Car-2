import matplotlib
import scipy
import numpy as np
from pyoctomap import OcTree


def indoor_map_gen():
    
    return
def outdoor_map_gen():
    return 


tree=OcTree(0.05)
tree.readBinary("/Users/tarunjaikumar/Documents/Robotics-UCSD/ERL-Racecar/catkin_ws/src/mushr_gazebo/exports/octree.bt")

res = tree.getResolution()
print("loaded octree: resolution =", res)

for leaf in tree.begin_leafs():
    coord = leaf.getCoordinate()
    if leaf.getSize() != res:
        print("pruned node at", coord, "size", leaf.getSize())


# convert the /cloud into the visualization msgs which is the ocotmap array markers

# .map files should turn into the octomap version of that file and there hsould be an octree built on it
 
 
#generate kdtrees from the .map



# the octree should also generate a obstacle distance function since you can connect the nodes and find the distances


# use pointnet to encode octree which I can then feed into the later neural network 



