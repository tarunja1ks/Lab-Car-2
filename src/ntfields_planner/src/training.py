import torch.nn as nn
import torch

# from octomap get the voxels for the obstacles and for free spaces

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
        return torch.maximum(qs,qg) 
        
class NTField:
    def __init__(self):
        print("thing")
        self.model=nn.Sequential(
            
        )
        
      
    
if __name__ == "__main__":
    print("inside main")