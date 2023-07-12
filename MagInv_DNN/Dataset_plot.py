""" 
This script  visualizes the simulated datase.
An  exapmle is randomly selected from datset followed by plotting its input and output. 
The input is magnetic anomaly and output is basement topography.

"""

#%%
import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.gridspec as gridspec
import json

#%%
""" SYNTHETIC MODELS DIMENTIONALITY AND PARAMETERS """
ntrain  = 10000
len_obs = 80.0
ndata = '10k'
nset = '1'

with open(f'N{ndata}_D{nset}.txt','r') as f:
    model_params = json.load(f)

y_fm = model_params['y_fm']
x_fm = model_params['x_fm']
z_fm = model_params['z_fm']
pad = model_params['pad']
tot_p = model_params['tot_p']
len_p = y_fm[1] - y_fm[0]
# lreg = np.arange(lmodel[0]-lp*pad, lmodel[-1] + lp*pad,lp)
num_ps = int((y_fm[-1] - y_fm[0])/len_p)              # number of prisms 

y_obs = np.arange(y_fm[0] + len_p/2,y_fm[-1] + len_p/2,len_p)
x_obs = (x_fm[1] - x_fm[0])/2
z_obs = 0.
nobs = np.shape(y_obs)[0]

#%%
pm=num_ps+1
# load the data
data = np.load(f"N{ndata}_D{nset}.npy")
# Choose a random number between 0 and number of training data
nd = random.sample(range(0,ntrain),k=1)
example = data[nd]
mag = example[:,:nobs]
mag = np.reshape(mag,nobs)
depth_base = example[:,nobs:]
depth_base = np.reshape(depth_base,tot_p)

dbase_plt = depth_base[pad:pm+pad]
#%%

xplt =  y_fm
fig1 = plt.figure(figsize=(8, 4.5))
gs0 = gridspec.GridSpec(2, 1, figure=fig1, height_ratios=[2, 3],hspace=0.2)

gs00 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs0[0],hspace=0.2)
ax1 = fig1.add_subplot(gs00[0])

ax1.plot(xplt[:-1],mag,c='r',linestyle='-',linewidth=1.1)

ax1.axis(xmin = xplt[0],xmax= xplt[-1])
# ax1.set_xlabel('xobs(km)',fontsize=8)
ax1.set_ylabel("DRTP-TF(nT)",fontsize=12)
# ax1.legend(loc='upper right',fontsize=10)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
gs01 = gs0[1].subgridspec(1, 1)
ax2 = fig1.add_subplot(gs01[0])
base = np.zeros((pm+3,2))
base[:pm,0] = xplt
base[:pm,1] = dbase_plt
base[-1,:] = [xplt[0],dbase_plt[0]]
base[-2,:] = [xplt[0],z_fm[1]]
base[-3,:] = [xplt[-1],z_fm[1]]

sediment = np.zeros((pm+3,2))
sediment[0,:] = [xplt[0],z_fm[0]]
sediment[1,:] = [xplt[-1],z_fm[0]]
sediment[-1,:] = [xplt[0],z_fm[0]]
sediment[2:pm+2,0] = np.flip(xplt)
sediment[2:pm+2,1] = np.flip(dbase_plt)

ax2.axis(xmin = xplt[0],xmax= xplt[-1])
ax2.set_ylim(z_fm[1],0)
# ax2.set_xlabel('X(km)')
ax2.set_ylabel("Depth(km)",fontsize=12)
ax2.set_xlabel('Distance(km)',fontsize=12)
# ax2.set_title('Prediction',loc='right',fontsize=8)
ax2.fill(base[:,0],base[:,1],facecolor='slategray', edgecolor='black', linewidth=0.3,alpha=1)
ax2.fill(sediment[:,0],sediment[:,1],facecolor= 'lightgrey', edgecolor='black', linewidth=0.3,label = "prediction")
# ax2.fill(base_sim[:,0],base_sim[:,1],facecolor='firebrick', edgecolor='black', linewidth=0.3,alpha=0.2)
# plt.setp(ax2.get_xticklabels(), visible=False)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# plt.savefig(f'SimFig{fign}.pdf') 
plt.show()