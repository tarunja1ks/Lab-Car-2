# import torch.nn as nn
# import torch

import mlx.core as mlx
import mlx.nn as nn

# code skeleton thing


#convert the .map files into octomap and make sure itll be portable into irl robot and the gazebo sim

# from octomap get the voxels for the obstacles and for free spaces

# no need to do the g(q) encoding and pass voxels for point cloud since its a mobile robot so the point cloud around it isnt too serious like an arm
# using the voxels pass through a neutral network of the feature maps and build the W matrix which is N*N*N*(k+1)

#raw configuration start

# combine these encoder values from qs and qg via the symmetric operator

# feed the combination into the neural network and train it

# build ground-truth speed field S*(q) from distance-to-obstacle

# loss formula 

# Adam optimizer

# produce a factorized time output

# decide the path to go on based on the output and sample numerous sourroundings similar to the mpc style loop
        
        
class Encoder():
    def __init__(self):
        self.Encoder_Model=nn.Sequential(
            nn.Linear(3,128), # going from the 3 feature tensor of x y theta into 128 to encode it
            nn.Softplus(),
            nn.Linear(128,128),
            nn.Softplus(),
        )
    
    def forward(self,q):
        return self.Encoder_Model(q)
    
    def symmetric_operator(self,qs,qg):
        return mlx.maximum(qs,qg) 
        
class NTField(nn.module):
    def __init__(self):
        self.model=nn.Sequential()
        
        self.constant_speed=20.0 # max speed place holder for now
        self.dmin=1
        self.dmax=0
        
        
    def speed_groundtruth(self,q):
        return self.constant_speed/self.dmax*mlx.clip(self.dmin,self.dmax)
    
        
    def eikonal_loss(self,QS_ground, QG_ground, QS_predict, QG_predict):
        return (mlx.abs(1-mlx.sqrt(QS_ground/QS_predict))+mlx.abs(1-mlx.sqrt(QG_ground/QG_predict))+mlx.abs(1-mlx.sqrt(QS_predict/QS_ground))+mlx.abs(1-mlx.sqrt(QG_predict/QG_ground))).mean()

    def compute_time(self,q):
        
        return 0
    
if __name__ == "__main__":
    
    epochs=1000
    for epoch in range(epochs):
        # do all the loss and minibatching
        