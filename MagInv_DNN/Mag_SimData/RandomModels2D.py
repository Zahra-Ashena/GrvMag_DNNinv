import numpy as np
import random
from scipy.stats import norm

class TopographyRand:
    def __init__(self):
        pass

    def topo_base_rand(self,**FM_params):

        """
            Returns an array of random values for the z-coordinate of the prisms (depth of first layer)  
        """   
        
        y_fm = FM_params['y_fm']
        pad = FM_params['pad']
        tot_p = FM_params['tot_p']
        num_anom = FM_params['num_anom']
        len_anom = FM_params['len_anom']
        topo = FM_params['topo']
        dep_avg = FM_params['dep_avg']
        w = FM_params['wsmooth'] 
        if_smooth = True

        len_p = y_fm[1] - y_fm[0]
        num_ps = int((y_fm[-1] - y_fm[0])/len_p)
            
        rnd_avg_dep = (dep_avg[1]-dep_avg[0]) * np.random.random_sample() + dep_avg[0]
        rnd_num_anom = random.choice(np.arange(num_anom[0],num_anom[1]))
    
        topo_base = np.zeros(tot_p,dtype=np.float16)
        avg_dep = rnd_avg_dep 
        topo_base [:] = avg_dep
        rnd_prisms_anom = random.choices(np.arange(len_anom[0],num_ps-len_anom[0]),k=rnd_num_anom) 
        rnd_prisms_anom= np.array(rnd_prisms_anom)
                            
        for prism in rnd_prisms_anom:

            rnd_anom_len = random.choice(range(len_anom[0],len_anom[1]))
            rnd_anom_rng = np.arange(prism-rnd_anom_len,prism+rnd_anom_len+1,len_p)
            rnd_anom_dep = (topo[1]-topo[0]) * np.random.random_sample() + topo[0] 
            anomaly = norm.pdf(rnd_anom_rng,prism,rnd_anom_dep)

            if rnd_anom_dep < avg_dep:
                anomaly=np.interp(anomaly, (anomaly.min(), anomaly.max()), (rnd_anom_dep,avg_dep))
                        
            else:
                anomaly=np.interp(anomaly, (anomaly.min(), anomaly.max()), (avg_dep,rnd_anom_dep))

            topo_base [pad+prism-rnd_anom_len:pad+prism+rnd_anom_len+1] = anomaly
        
        if if_smooth==True:
            topo_base = np.convolve(topo_base, np.ones(w),'same') / w
            edg = int(np.floor(w/2))
            topo_base[:edg] = topo_base[edg+1]
            topo_base[-edg:] = topo_base[-edg-1]
                    
        return topo_base



    
             
  

    

    