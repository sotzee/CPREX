# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 18:03:02 2016

@author: sotzee
"""

import scipy.constants as const
from scipy.constants import physical_constants
import numpy as np

c=100*const.c
G=1000*const.G
e=1e7*const.e
kB=const.k*1e7
sigma_sb=const.Stefan_Boltzmann*1e3
hbar=const.hbar*1e7
mass_per_baryon=const.m_n*1000
m_p_MeV=const.m_p*const.c**2/(1e6*const.e)
m_n_MeV=const.m_n*const.c**2/(1e6*const.e)
m_e_MeV=physical_constants['electron mass'][0]*const.c**2/(1e6*const.e)
m_mu_MeV=physical_constants['muon mass'][0]*const.c**2/(1e6*const.e)
#m_e_MeV/=10000  #massless electron
#m_mu_MeV*=10000 #without muon
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
m_muon_MeV=m_mu_MeV
MeV_fm=(1e6*e/hbar/c)/1e13
unitMeVfm=(MeV_fm)**3
unitPressure=(1e6*e)**4/(hbar*c)**3
unitDensity=(1e6*e)**4/(hbar*c)**3/c**2
unitBaryonDensity=1e-39
unitTemperature_from_MeV=(1e6*e)/kB
unitTemperature_from_fm=1e15*const.e**2/(4*np.pi*const.epsilon_0*const.k)
unitKappa=unitMeVfm**(2/3)*1e26*c*kB
Gc2=G/c**2
fine_constant=const.e**2/(4*np.pi*const.epsilon_0*const.hbar*const.c)


def toPressure(pressure_before,unit_before):
    if(unit_before=='mevfm' or unit_before=='mevfm3' or unit_before=='mevfm-3'):
        return pressure_before/unitMeVfm*unitPressure
    if(unit_before=='mev4'):
        return pressure_before*unitPressure
    if(unit_before=='mev'):
        return pressure_before**4.0*unitPressure
    if(unit_before=='fm-4'):
        return pressure_before/unitMeVfm**(4./3)*unitPressure
        
def toDensity(density_before,unit_before):
    if(unit_before=='mevfm' or unit_before=='mevfm3' or unit_before=='mevfm-3'):
        return density_before/unitMeVfm*unitDensity
    if(unit_before=='mev4'):
        return density_before*unitDensity
    if(unit_before=='mev'):
        return density_before**4.0*unitDensity
    if(unit_before=='fm-4'):
        return density_before/unitMeVfm**(4./3)*unitDensity

def toBaryonDensity(density_before,unit_before):
    if(unit_before=='mevfm' or unit_before=='mevfm3' or unit_before=='mevfm-3'):
        return density_before/unitBaryonDensity

def toTemperature(temperature_before,unit_before):
    if(unit_before=='mev' or unit_before=='Mev' or unit_before=='MeV'):
        return temperature_before*unitTemperature_from_MeV
    if(unit_before=='fm' or unit_before=='fm-1'): #via k_e*e^2/lenth=k_B T, not via hbar*c/lenth=k_B T   
        return temperature_before*unitTemperature_from_fm

def toLength(length_before,unit_before):
    if(unit_before=='mev-1fm3_sqr'):
        return length_before*(unitDensity*Gc2/unitMeVfm)**(-0.5)

def toMevfm(before,unit_before):
    if(unit_before=='pressure'):
        return before*unitMeVfm/unitPressure
    if(unit_before=='density'):
        return before*unitMeVfm/unitDensity
    if(unit_before=='baryondensity'):
        return before*unitBaryonDensity
    if(unit_before=='mev4'):
        return before*unitMeVfm
    if(unit_before=='km-2'):
        return before*unitMeVfm/(unitDensity*Gc2*1e10)

def toMev4(before,unit_before):
    if(unit_before=='pressure'):
        return before/unitPressure
    if(unit_before=='density'):
        return before/unitDensity
    if(unit_before=='mevfm' or unit_before=='mevfm3' or unit_before=='mevfm-3'):
        return before/unitMeVfm
        
def toMev(before,unit_before):
    if(unit_before=='pressure'):
        return (before/unitPressure)**0.25
    if(unit_before=='density'):
        return (before/unitDensity)**0.25
    if(unit_before=='temperature'):
        return before/unitTemperature_from_MeV
        
#test:
#print('test:')
#print(toMevfm(toPressure(10,'mevfm'),'pressure'))
#print(toMevfm(toDensity(10,'mevfm'),'density'))
#print(toMev4(toPressure(10,'mev4'),'pressure'))
#print(toMev4(toDensity(10,'mev4'),'density'))
#print(toMev(toPressure(10,'mev'),'pressure'))
#print(toMev(toDensity(10,'mev'),'density'))