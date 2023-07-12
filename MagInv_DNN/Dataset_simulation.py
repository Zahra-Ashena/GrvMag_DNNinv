"""
Magnetic Training Data Set Simulation
=====================================

This script simulates magnetic training data sets by designing a forward model and setting its dimensions.
It assigns the physical and structural properties to the forward model to generate synthetic magnetic data.
The forward model represents a mathematical model that relates the physical properties of a system to its
observed magnetic behavior. By simulating magnetic data, this script allows for testing and training 
machine learning models on synthetic data before applying them to real-world magnetic data sets.

The training dat set is composed og magnetic anomaly as input and basement topography as output.

Author: Zahra Ashena
Date: July 2023

"""

# %%
import numpy as np
from ForwardModel2D_MaG import MAGNETIC
from RandomModels2D import TopographyRand
import json
import multiprocessing 

# %%
MAG = MAGNETIC()
TR = TopographyRand()

# %% 
# Select the lenght of trainig set to be simulated (e.g. ntrain = 100000)
# and the lenght of the observation profile (same as lenght of the forward model) in kilometers
# e.g. len_fm = 100.0
ntrain  = 10000
len_obs = 80.0
ndata = '10k'
nset = '1'

# %%
# Forward model dimentions and coordinates 
len_p = 1.0                                           # lenght of each prism in km
len_fm = len_obs                                      # lenght of the forward model in km
wd_fm = 20*len_p                                      # width of the forward model in km
y_fm = np.arange(-len_fm/2,(len_fm/2)+len_p,len_p)
x_fm = np.array([0.0,wd_fm])
z_fm = np.array([0.0,12.0])
num_ps = int((y_fm[-1] - y_fm[0])/len_p)              # number of prisms 

# %%
# Observation coordinates and main field parameters
y_obs = np.arange(y_fm[0] + len_p/2,y_fm[-1] + len_p/2,len_p)
x_obs = (x_fm[1] - x_fm[0])/2
z_obs = 0.
nobs = np.shape(y_obs)[0]
inc_obs = 90.                                                # inclination of the observation
dec_obs = 0.                                                 # declination of the observation

# %% 
# Structural parameters of the forward model
num_anom = np.array([1,8])                                   # min and max number of anomalies
len_anom = np.array([4,20])                                  # min and max length of an anomaly in km
dep_avg  = np.array([5.0,9.0])                               # the range to choose the average depth of the basement 
topo = np.array([3.0,11.0])                                  # the range to choose the topography of the basement 
pad = len_anom[1]*2                                          # number of prisms to be padded to the model to avoid edge effects
tot_p = num_ps + 2*pad                                       # total number of prisms

# %% 
# Physical and Main field parameters of the forward model
inc_p = np.ones(tot_p) * 90.                                # inclination of the prisms
dec_p = np.ones(tot_p) * 0.                                 # declination of the prisms
K = np.ones(tot_p)*0.005                                    # susceptiblity of the prisms

# %% 
# Save variables into a dictionary
if_smooth = True                                           # to smooth the totpography 
w_smooth = 10                                              # the more the smother topography

FM_params = {'y_fm': y_fm.tolist(),'x_fm':x_fm.tolist(),
                    'z_fm': z_fm.tolist(), 'num_anom': num_anom.tolist(),
                    'len_anom':len_anom.tolist(),'pad':int(pad),'tot_p': int(tot_p),'x_obs':x_obs,'y_obs': y_obs.tolist(),
                    'z_obs':z_obs,'K': K.tolist(), 'inc_obs': inc_obs,'dec_obs': dec_obs, 'inc_p':inc_p.tolist(),'dec_p': dec_p.tolist(),'num_train':ntrain,
                    'topo': topo.tolist(),'dep_avg': dep_avg.tolist(),'wsmooth': w_smooth}

with open(f'N{ndata}_D{nset}.txt','w') as f:
    f.write(json.dumps(FM_params)) 

# %%
y_reg = np.arange(y_fm[0] - len_p*pad, y_fm[-1] + len_p*pad,len_p)
Zp_dw = np.repeat(z_fm[1],tot_p)
lx = x_fm[1]-x_fm[0]
ly = y_fm[1]-y_fm[0]
Yp_c = y_reg + len_p/2
Xp_c = np.repeat(lx/2,tot_p)
depth_base = TR.topo_base_rand(**FM_params)


# %%
# # Dataset simulation:
# Given the number of data, in each iteration a random basement topography is created. Then the magnetic anomaly
# of the basement is calcuted. The final training set is composed of magnetic anomaly of each forward model coupled 
# with its basement topography
def ParForward(i):

    depth_base = TR.topo_base_rand(**FM_params)
    Zp_up = depth_base
    mag = np.zeros(nobs)
    for j in range(nobs):
        mag1 = MAG.MaG_Layer2D(lx, ly, x_obs, y_obs[j], Xp_c, Yp_c, Zp_up, Zp_dw, inc_obs, dec_obs, inc_p, dec_p)
        mag[j] = np.dot(mag1,K) 

    return np.append(mag,depth_base)

# %%
if __name__=='__main__':

    cpunumbers = multiprocessing.cpu_count()
            
    pro = multiprocessing.Pool(cpunumbers-2)
    train_set_fin = pro.map(ParForward, np.arange(ntrain))
            
    pro.close()
    pro.join()
    
    np.save(f"N{ndata}_D{nset}.npy", train_set_fin)


