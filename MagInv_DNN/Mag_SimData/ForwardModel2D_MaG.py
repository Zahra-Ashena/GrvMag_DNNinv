from math import *
import numpy as np
from decimal import *
import random
from itertools import *
import math


class MAGNETIC:
    def __init__(self):
        pass

    """
        A class for calculating magnetic anomalies:

        Methods:
        (1) "MaG_Prsm2D"  - Calculates the magnetic anomaly of a single prism at an observation point.
        (2) "MaG_Layer2D" - Calculates the magnetic anomaly of an ensemble of prisms at an observation point.
            
    """

    def MaG_Prsm2D(lx, ly, lz, xobs, yobs, xp_c, yp_c, zp_c, inc_obs, dec_obs, inc_p, dec_p):

        """
            Calculates the magnetic anomaly of a prism in nanoTesla at one observation point:
            
            Inputs:
            lx,      ly,   lz  : Width, length and height of the prism in kilometers
            xp_c,    yp_c, zp_c:  x, y and z coordinate of the center of the prism
            inc_p,   dec_p     : Inclination and declination of the prism in degrees
            xobs,    yobs      : Coordinates of the observation point in kilometers
            inc_obs, dec_obs   : Inclination and Declination  of the observation point in degrees

            Note:
            - The units of the inputs are degrees for angles and kilometers for distances.

        """

        km2cm = 1e5
        lx = lx*km2cm
        ly = ly*km2cm
        lz = lz*km2cm
        xobs = xobs*km2cm
        yobs = yobs*km2cm
        yp_c = yp_c*km2cm
        xp_c = xp_c*km2cm
        inc_obs = inc_obs*math.pi/180
        dec_obs = dec_obs*math.pi/180
        inc_p = inc_p*math.pi/180
        dec_p = dec_p*math.pi/180

        p = math.cos(inc_obs)*math.cos(dec_obs); q=math.cos(inc_obs)*math.sin(dec_obs); r=math.sin(inc_obs)
        L = math.cos(inc_p)*math.cos(dec_p); M=math.cos(inc_p)*math.sin(dec_p); N=math.sin(inc_p)
        G1 = (M*r+N*q); G2=(L*r+N*p); G3=(L*q+M*p); G4=(N*r-M*q); G5=(N*r-L*p)

        Xj = [xp_c-lx/2 , xp_c+lx/2]
        Yj = [yp_c-ly/2 , xp_c+ly/2]

        ak0 = Xj[0] - xobs
        ak1 = Xj[1] - xobs
        bl0 = Yj[0] - yobs
        bl1 = Yj[1] - yobs
        cm0 = zp_c
        cm1 = lz + zp_c

        Rklm1 = math.sqrt(ak0**2+bl0**2+cm0**2)
        ww1 = -(G1*math.log(Rklm1+ak0)+G2*math.log(Rklm1+bl0)+G3*math.log(Rklm1+cm0)+G4*math.atan((ak0*cm0)/(Rklm1*bl0))+G5*math.atan((bl0*cm0)/(Rklm1*ak0)))
        Rklm2 = math.sqrt(ak0**2+bl0**2+cm1**2)
        ww2 = (G1*math.log(Rklm2+ak0)+G2*math.log(Rklm2+bl0)+G3*math.log(Rklm2+cm1)+G4*math.atan((ak0*cm1)/(Rklm2*bl0))+G5*math.atan((bl0*cm1)/(Rklm2*ak0)))
        Rklm3 = math.sqrt(ak0**2+bl1**2+cm0**2)
        ww3 = (G1*math.log(Rklm3+ak0)+G2*math.log(Rklm3+bl1)+G3*math.log(Rklm3+cm0)+G4*math.atan((ak0*cm0)/(Rklm3*bl1))+G5*math.atan((bl1*cm0)/(Rklm3*ak0)))
        Rklm4 = math.sqrt(ak0**2+bl1**2+cm1**2)
        ww4 = -(G1*math.log(Rklm4+ak0)+G2*math.log(Rklm4+bl1)+G3*math.log(Rklm4+cm1)+G4*math.atan((ak0*cm1)/(Rklm4*bl1))+G5*math.atan((bl1*cm1)/(Rklm4*ak0)))
        Rklm5 = math.sqrt(ak1**2+bl0**2+cm0**2)
        ww5 = (G1*math.log(Rklm5+ak1)+G2*math.log(Rklm5+bl0)+G3*math.log(Rklm5+cm0)+G4*math.atan((ak1*cm0)/(Rklm5*bl0))+G5*math.atan((bl0*cm0)/(Rklm5*ak1)))
        Rklm6 = math.sqrt(ak1**2+bl0**2+cm1**2)
        ww6 = -(G1*math.log(Rklm6+ak1)+G2*math.log(Rklm6+bl0)+G3*math.log(Rklm6+cm1)+G4*math.atan((ak1*cm1)/(Rklm6*bl0))+G5*math.atan((bl0*cm1)/(Rklm6*ak1)))
        Rklm7 = math.sqrt(ak1**2+bl1**2+cm0**2)
        ww7 = -(G1*math.log(Rklm7+ak1)+G2*math.log(Rklm7+bl1)+G3*math.log(Rklm7+cm0)+G4*math.atan((ak1*cm0)/(Rklm7*bl1))+G5*math.atan((bl1*cm0)/(Rklm7*ak1)))
        Rklm8 = math.sqrt(ak1**2+bl1**2+cm1**2)
        ww8 = (G1*math.log(Rklm8+ak1)+G2*math.log(Rklm8+bl1)+G3*math.log(Rklm8+cm1)+G4*math.atan((ak1*cm1)/(Rklm8*bl1))+G5*math.atan((bl1*cm1)/(Rklm8*ak1)))
        Gmag_f = ww1+ww2+ww3+ww4+ww5+ww6+ww7+ww8

        return Gmag_f*1e5
       
    
    def MaG_Layer2D(self,lx, ly, xobs, yobs, Xp_c, Yp_c, Zp_up, Zp_dw, inc_obs, dec_obs, inc_ps, dec_ps):
        
        """
            Calculates the magnetic anomaly of an ensemble of prisms in nanoTesla at one 
            observation point:

            Inputs:
            lx,      ly     : Width and length of each prism in kilometers
            Xp_c,    Yp_c   : Arrays containing the x and y coordinates of the center of the prisms
            Zp_up,   Zp_dw  : Arrays containing the upper depths and lower depths of the prisms in kilometers
            inc_b,   dec_b  : Arrays containing the inclination and declination of the prisms in degrees
            xobs,    yobs   : Coordinates of the observation point in kilometers
            inc_obs, dec_obs: Inclination and Declinationof the observation point in degrees

            Note:
            - The units of the inputs are degrees for angles and kilometers for distances.
            - The arrays Xp_c, Yp_c, Zp_up, Zp_dw, inc_b, and dec_b should have the same length,
            representing the properties of individual prisms in the ensemble.

        """

        km2cm = 1e5
        lx = lx*km2cm
        ly = ly*km2cm
        xobs = xobs*km2cm
        yobs = yobs*km2cm
        Yp_c = Yp_c*km2cm
        Xp_c = Xp_c*km2cm
        Zp_up = Zp_up*km2cm
        Zp_dw = Zp_dw*km2cm
        
        inc_obs=inc_obs*math.pi/180
        dec_obs=dec_obs*math.pi/180
        inc_ps=inc_obs*math.pi/180
        dec_ps=dec_obs*math.pi/180
        
        p=math.cos(inc_obs)*math.cos(dec_obs); q=math.cos(inc_obs)*math.sin(dec_obs); r=math.sin(inc_obs)
        L=np.cos(inc_ps)*np.cos(dec_ps); M=np.cos(inc_ps)*np.sin(dec_ps); N=np.sin(inc_ps)
        G1=(M*r+N*q); G2=(L*r+N*p); G3=(L*q+M*p); G4=(N*r-M*q); G5=(N*r-L*p)
        Xj=[Xp_c-lx/2 , Xp_c+lx/2]
        Yj=[Yp_c-ly/2 , Yp_c+ly/2]

        ak0=Xj[:][0] - xobs
        ak1=Xj[:][1] - xobs
        bl0=Yj[:][0] - yobs
        bl1=Yj[:][1] - yobs
        cm0 = Zp_up
        cm1 = Zp_dw
       
        Rklm1=np.sqrt(ak0**2+bl0**2+cm0**2)
        ww1=-(G1*np.log(Rklm1+ak0)+G2*np.log(Rklm1+bl0)+G3*np.log(Rklm1+cm0)+G4*np.arctan((ak0*cm0)/(Rklm1*bl0))+G5*np.arctan((bl0*cm0)/(Rklm1*ak0)))
        Rklm2=np.sqrt(ak0**2+bl0**2+cm1**2)
        ww2=(G1*np.log(Rklm2+ak0)+G2*np.log(Rklm2+bl0)+G3*np.log(Rklm2+cm1)+G4*np.arctan((ak0*cm1)/(Rklm2*bl0))+G5*np.arctan((bl0*cm1)/(Rklm2*ak0)))
        Rklm3=np.sqrt(ak0**2+bl1**2+cm0**2)
        ww3=(G1*np.log(Rklm3+ak0)+G2*np.log(Rklm3+bl1)+G3*np.log(Rklm3+cm0)+G4*np.arctan((ak0*cm0)/(Rklm3*bl1))+G5*np.arctan((bl1*cm0)/(Rklm3*ak0)))
        Rklm4=np.sqrt(ak0**2+bl1**2+cm1**2)
        ww4=-(G1*np.log(Rklm4+ak0)+G2*np.log(Rklm4+bl1)+G3*np.log(Rklm4+cm1)+G4*np.arctan((ak0*cm1)/(Rklm4*bl1))+G5*np.arctan((bl1*cm1)/(Rklm4*ak0)))
        Rklm5=np.sqrt(ak1**2+bl0**2+cm0**2)
        ww5=(G1*np.log(Rklm5+ak1)+G2*np.log(Rklm5+bl0)+G3*np.log(Rklm5+cm0)+G4*np.arctan((ak1*cm0)/(Rklm5*bl0))+G5*np.arctan((bl0*cm0)/(Rklm5*ak1)))
        Rklm6=np.sqrt(ak1**2+bl0**2+cm1**2)
        ww6=-(G1*np.log(Rklm6+ak1)+G2*np.log(Rklm6+bl0)+G3*np.log(Rklm6+cm1)+G4*np.arctan((ak1*cm1)/(Rklm6*bl0))+G5*np.arctan((bl0*cm1)/(Rklm6*ak1)))
        Rklm7=np.sqrt(ak1**2+bl1**2+cm0**2)
        ww7=-(G1*np.log(Rklm7+ak1)+G2*np.log(Rklm7+bl1)+G3*np.log(Rklm7+cm0)+G4*np.arctan((ak1*cm0)/(Rklm7*bl1))+G5*np.arctan((bl1*cm0)/(Rklm7*ak1)))
        Rklm8=np.sqrt(ak1**2+bl1**2+cm1**2)
        ww8=(G1*np.log(Rklm8+ak1)+G2*np.log(Rklm8+bl1)+G3*np.log(Rklm8+cm1)+G4*np.arctan((ak1*cm1)/(Rklm8*bl1))+G5*np.arctan((bl1*cm1)/(Rklm8*ak1)))
        
        Gmag_f=ww1+ww2+ww3+ww4+ww5+ww6+ww7+ww8

        return Gmag_f*1e5





    


   


    