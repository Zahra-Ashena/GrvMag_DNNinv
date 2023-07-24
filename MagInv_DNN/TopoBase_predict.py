"""
Prediction of Basement Topography using Trained DNN Model
========================================================

This script uses a trained DNN model to predict the basement topography based on unseen magnetic data. The DNN model has been previously trained on simulated magnetic data sets and is now applied to new magnetic data.

Author: Zahra Ashena;
Date: July 2023

"""
# %%
from tensorflow import keras
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from Mag_SimData.ForwardModel2D_MaG import MAGNETIC
from Mag_SimData.RandomModels2D import TopographyRand
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
import matplotlib.gridspec as gridspec
import json

MAG = MAGNETIC()
TR = TopographyRand()
# scaler = MinMaxScaler()
# norm = Normalizer()

# %%
# Load the forward model parameters

len_obs = 80.0
ndata = '10k'
nset = '1'
nexp = '1'
ns = '0%'
noise = 0.1
nfig ='1'

with open(f'N{ndata}_D{nset}.txt','r') as f:
    FM_params  = json.load(f)

y_fm = FM_params['y_fm']
x_fm = FM_params['x_fm']
z_fm = FM_params['z_fm']
y_obs = np.array(FM_params['y_obs'])
x_obs = FM_params['x_obs']
pad = FM_params['pad']
tot_p = FM_params['tot_p']
w = FM_params['wsmooth']
len_p = y_fm[1] - y_fm[0]
num_ps = int((y_fm[-1] - y_fm[0])/len_p)  
K = FM_params['K']
inc_obs = FM_params['inc_obs']
dec_obs = FM_params['dec_obs']
inc_p = np.array(FM_params['inc_p'])
dec_p = np.array(FM_params['dec_p'])

# %%
# Load the trained DNN model

DNN_model = keras.models.load_model(f'N{ndata}_D{nset}_M{nexp}.h5')

# %%
# Create an unseen synthetic data

topobase_sim = TR.topo_base_rand(**FM_params)
y_reg = np.arange(y_fm[0] - len_p*pad, y_fm[-1] + len_p*pad,len_p)
lx = x_fm[1]-x_fm[0]
ly = y_fm[1]-y_fm[0]
Yp_c = y_reg + len_p/2
Xp_c = np.repeat(lx/2,tot_p)
Zp_dw = np.repeat(z_fm[1],tot_p)
Zp_up = topobase_sim
nobs = np.shape(y_obs)[0]      
mag_sim = np.zeros(nobs)
for i in range(nobs):
    magi = MAG.MaG_Layer2D(lx, ly, x_obs, y_obs[i], Xp_c, Yp_c, Zp_up, Zp_dw, inc_obs, dec_obs, inc_p, dec_p)
    mag_sim[i] = np.dot(magi,K) 


# %%
# Adding noise to the synthetic unseen data
# mag_sim = np.add(mag_sim,np.random.normal(0,mag_sim.std(), nobs)*noise)

# %%
# DNN model prediction

topobase_pred = DNN_model.predict(np.reshape(mag_sim, (-1,nobs)))
topobase_pred = np.reshape(topobase_pred,(tot_p))

# %%
# Calculate the magnetic anomaly of the predicted basement

mag_pred = np.zeros(nobs)
Zp_up_pre = topobase_pred
mag_pred = np.zeros(nobs)
for j in range(nobs):
    magj = MAG.MaG_Layer2D(lx, ly, x_obs, y_obs[j], Xp_c, Yp_c, Zp_up_pre, Zp_dw, inc_obs, dec_obs, inc_p, dec_p)
    mag_pred[j] = np.dot(magj,K) 

# mag_pred = np.reshape(mag_pred,nobs)

# Smooth the predicted topography
topobase_pred = np.convolve(topobase_pred, np.ones(w),'same') / w
edg = int(np.floor(w/2))
topobase_pred[:edg] = topobase_pred[edg+1]
topobase_pred[-edg:] = topobase_pred[-edg-1]

test_mse = mean_squared_error(topobase_sim,topobase_pred)
print ("Test MSE: ", test_mse)

# %%
# Plot the results

pm = num_ps+1
dbase_sim_plt = topobase_sim[pad:pm+pad]
dbase_pred_plt = topobase_pred[pad:pm+pad]
xplt = y_fm
# xplt =  np.arange(-len_obs/2,(len_obs/2) + len_p ,len_p)

fig1 = plt.figure(figsize=(8, 4.4))
gs0 = gridspec.GridSpec(2, 1, figure=fig1, height_ratios=[2, 3],hspace=0.2)

gs00 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs0[0],hspace=0.2)
ax1 = fig1.add_subplot(gs00[0])

ax1.plot(xplt[:-1],mag_sim,c='r',linestyle='-',linewidth=1.1,label = "Simulation")
ax1.plot(xplt[:-1],mag_pred,c='blue',linestyle='--',linewidth=1.1,label = "Prediction")

ax1.axis(xmin = xplt[0],xmax= xplt[-1])
# ax1.set_xlabel('xobs(km)',fontsize=8)
ax1.set_ylabel("DRTP-TF(nT)",fontsize=10)
ax1.legend(loc='upper right',fontsize=10)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
gs01 = gs0[1].subgridspec(1, 1)
ax2 = fig1.add_subplot(gs01[0])
base_pred = np.zeros((pm+3,2))
base_pred[:pm,0] = xplt
base_pred[:pm,1] = dbase_pred_plt
base_pred[-1,:] = [xplt[0],dbase_pred_plt[0]]
base_pred[-2,:] = [xplt[0],z_fm[1]]
base_pred[-3,:] = [xplt[-1],z_fm[1]]
base_sim = np.zeros((pm+3,2))
base_sim[:pm,0] = xplt
base_sim[:pm,1] = dbase_sim_plt
base_sim[-1,:] = [xplt[0],dbase_sim_plt[0]]
base_sim[-2,:] = [xplt[0],z_fm[1]]
base_sim[-3,:] = [xplt[-1],z_fm[1]]

sed_pred = np.zeros((pm+3,2))
sed_pred[0,:] = [xplt[0],z_fm[0]]
sed_pred[1,:] = [xplt[-1],z_fm[0]]
sed_pred[-1,:] = [xplt[0],z_fm[0]]
sed_pred[2:pm+2,0] = np.flip(xplt)
sed_pred[2:pm+2,1] = np.flip(dbase_pred_plt)

ax2.axis(xmin = xplt[0],xmax= xplt[-1])
ax2.set_ylim(z_fm[1],0)
# ax2.set_xlabel('X(km)')
ax2.set_ylabel("Depth(km)",fontsize=12)
ax2.set_xlabel('Distance(km)',fontsize=12)
# ax2.set_title('Prediction',loc='right',fontsize=8)
ax2.fill(base_pred[:,0],base_pred[:,1],facecolor='slategray', edgecolor='black', linewidth=0.3,alpha=1)
ax2.fill(sed_pred[:,0],sed_pred[:,1],facecolor= 'lightgrey', edgecolor='black', linewidth=0.3,label = "prediction")

# ax2.fill(base_sim[:,0],base_sim[:,1],facecolor='firebrick', edgecolor='black', linewidth=0.3,alpha=0.2)
ax2.plot(xplt,dbase_sim_plt,'w',linestyle='--',linewidth=1.7 )

# plt.setp(ax2.get_xticklabels(), visible=False)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig(f'PredFig{nfig}_ns{ns}.pdf') 
plt.show()

# %% [markdown]
# Save synthetic and prediction data

# %%
syn_pred = {'mag_sim': mag_sim.tolist(),'mag_pred': mag_pred.tolist(),
           'topobase_syn':topobase_sim.tolist(),'topobase_pred': topobase_pred.tolist(), 
           'TestMSE':test_mse}

with open(f'FigPred{nfig}_ns{ns}.txt','w') as f:
    f.write(json.dumps(syn_pred)) 



