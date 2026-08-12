# import torch.nn as nn
# import torch

import mlx.core as mlx
import mlx.nn as nn
import mlx.optimizers as optimizer
import numpy as np
from pyoctomap import OcTree
from scipy.spatial import cKDTree
from tqdm import tqdm
import time

# the eikonal loss differentiates through grad(T), and mlx's fused/compiled
mlx.disable_compile()



# dataset generation i asked claude to make 
def load_free_points(path, lidar_z=None, z_tol=0.1):
    data = np.loadtxt(path)
    if lidar_z is not None:
        data = data[np.abs(data[:, 2] - lidar_z) < z_tol]
    return data[:, :2], data[:, 3]      # xy, sizes


def sample_positions(free_xy, free_sizes, n):
    idx = np.random.randint(0, len(free_xy), n)
    half = (free_sizes[idx] / 2.0)[:, None]
    jitter = np.random.uniform(-1.0, 1.0, (n, 2)) * half
    return free_xy[idx] + jitter


# nt fields class         
class NTField(nn.Module):
    def __init__(self,kdtree):
        super().__init__()
        self.f_encoder=nn.Sequential(
            nn.Linear(2, 128), nn.Softplus(),   
            nn.Linear(128, 128), nn.Softplus(),
        )
        self.tao_model=nn.Sequential(
            nn.Linear(256, 128), nn.Softplus(),
            nn.Linear(128, 1), nn.Sigmoid(),
        )
        self.constant_speed=1.0 
        self.dmin=0.2
        self.dmax=10
        self.kdtree=kdtree
        
    def symmetric_operator(self,f_qs,f_qg):
        return mlx.concatenate([mlx.maximum(f_qs,f_qg) ,mlx.minimum(f_qs,f_qg)],axis=-1)
    
    def speed_groundtruth(self,q):
        d, index = self.kdtree.query(np.array(q)[:, :2])
        d = mlx.array(d).reshape(-1, 1)
        return self.constant_speed/self.dmax*mlx.clip(d,self.dmin,self.dmax)
        
    def eiknal_loss(self,QS_ground, QG_ground, QS_predict, QG_predict):
        return (mlx.abs(1-mlx.sqrt(QS_ground/QS_predict))+mlx.abs(1-mlx.sqrt(QG_ground/QG_predict))+mlx.abs(1-mlx.sqrt(QS_predict/QS_ground))+mlx.abs(1-mlx.sqrt(QG_predict/QG_ground))).mean()

        
    def speed_recovery(self,qs,qg):
        def norm(v):
            return mlx.sqrt(mlx.sum(v * v, axis=-1, keepdims=True) + 1e-12) # added a small value so norm wont be 0

        def T_of_qs(x):
            r=norm(qg-x)
            tao=self.tao_model(self.symmetric_operator(self.f_encoder(x), self.f_encoder(qg)))
            return mlx.sum(r / tao)

        def T_of_qg(x):
            r=norm(x-qs)
            tao=self.tao_model(self.symmetric_operator(self.f_encoder(qs), self.f_encoder(x)))
            return mlx.sum(r/tao)

        qs_grad_T=mlx.grad(T_of_qs)(qs)
        ss=1/mlx.maximum(1e-8, norm(qs_grad_T))

        qg_grad_T=mlx.grad(T_of_qg)(qg)
        sg=1/mlx.maximum(1e-8, norm(qg_grad_T))

        return ss,sg
    
    

        
    
if __name__ == "__main__":
    
    
    # read in the kdtree of the map
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
    points = np.unique(np.concatenate(all_points, axis=0), axis=0)
    points=cKDTree(points)
    
    
    
    
    # instaniate the NTFields
    ntfield=NTField(points)
    
    # getting the sampled points
    free_xy, free_sizes = load_free_points("/Users/tarunjaikumar/Documents/Robotics-UCSD/ERL-Racecar/catkin_ws/src/mushr_gazebo/exports/free.txt")
    qs = sample_positions(free_xy, free_sizes, 200000)
    qg = sample_positions(free_xy, free_sizes, 200000)
    
    epochs=3000
    batchsize=512
    opt = optimizer.AdamW(learning_rate=1e-4)          
            
            
    loss_values=[]
    for epoch in tqdm(range(epochs),desc="epoch completed"):
        # do all the loss and minibatching
        perm=np.random.permutation(len(qs))
        total, steps = 0.0, 0
        for i in range(0, len(qs), batchsize):      
            idx = perm[i:i+batchsize]
            
            # converting from numpy into mlx
            qs_b = mlx.array(qs[idx].astype(np.float32))
            qg_b = mlx.array(qg[idx].astype(np.float32))

            
            # precomputing the groundtruths
            ss_star = ntfield.speed_groundtruth(qs_b)
            sg_star = ntfield.speed_groundtruth(qg_b)

            def loss_fn(model):
                S_s, S_g = model.speed_recovery(qs_b, qg_b)
                return model.eiknal_loss(ss_star, sg_star, S_s, S_g)

            loss_and_grad_fn = nn.value_and_grad(ntfield, loss_fn)
            loss, grads = loss_and_grad_fn(ntfield)
            opt.update(ntfield, grads)
            mlx.eval(ntfield.parameters(), opt.state)
            
            
            loss_values.append(float(loss))
            total += loss_values[-1]
            steps += 1
        # print(total/steps, " is the current loss for this batch")
    print(total/steps,"is the loss overall now")
    np.savetxt("losses.txt", np.array(loss_values))
    ntfield.f_encoder.save_weights("f_encoder.npz")
    ntfield.tao_model.save_weights("tao.npz")
            
            
            
        

 
