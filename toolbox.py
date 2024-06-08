#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 14:09:01 2019

@author: sotzee
"""
import pickle
import numpy as np
import os
import matplotlib as mpl

linestyle_list_6=['-','-.','--',':',(0, (3, 1, 1, 1, 1, 1)),(0, (3, 5, 1, 5)),(0, (3, 5, 1, 5, 1, 5))]
color_list_10=['tab:orange','tab:blue','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']
color_list_15=color_list_10+['m','lime','navy','yellow','salmon']
color_list_20=color_list_15+['indigo','saddlebrown','darkslategrey','darkgoldenrod','darkolivegreen']

tab20=mpl.colormaps['tab20']

circle = mpl.path.Path.unit_circle()
verts = np.copy(circle.vertices)
verts[:, 0] *= 1.5
ellipse_marker_fat = mpl.path.Path(np.copy(verts), circle.codes)
verts[:, 0] *= 2
ellipse_marker = mpl.path.Path(np.copy(verts), circle.codes)
verts[:, 0] *= 2
ellipse_marker_thin = mpl.path.Path(np.copy(verts), circle.codes)
#use with the args: marker=ellipse_marker

def ensure_dir(path,dir_name):
    try:
        os.stat(path+dir_name)
    except:
        os.mkdir(path+dir_name)

def pickle_load(path,dir_name,data_name_list):
    data=[]
    for data_name_i in data_name_list:
        f=open(path+dir_name+'/'+data_name_i+'.dat','rb')
        data.append(pickle.load(f))
        f.close()
    return data

def pickle_dump(path,dir_name,tuple_data_dataname):
    for data_i,data_name_i in tuple_data_dataname:
        f=open(path+dir_name+'/'+data_name_i+'.dat','wb')
        pickle.dump(data_i,f)
        f.close()
        
def round_sig(x,sig=4):
    x = float(np.format_float_positional(x, precision=sig, unique=False, fractional=False,trim='k'))
    return x
round_sig = np.vectorize(round_sig)

def swap(array,index1,index2):
    array[index1],array[index2]=array[index2],array[index1]
    return array

def build_args(limit_list,shape):
    if(len(limit_list)==len(shape)):
        if(len(shape)==1):
            return np.mgrid[limit_list[0][0]:limit_list[0][1]:shape[0]*1j]
        elif(len(shape)==2):
            return np.mgrid[limit_list[0][0]:limit_list[0][1]:shape[0]*1j,limit_list[1][0]:limit_list[1][1]:shape[1]*1j]
        elif(len(shape)==3):
            return np.mgrid[limit_list[0][0]:limit_list[0][1]:shape[0]*1j,limit_list[1][0]:limit_list[1][1]:shape[1]*1j,limit_list[2][0]:limit_list[2][1]:shape[2]*1j]
        elif(len(shape)==4):
            return np.mgrid[limit_list[0][0]:limit_list[0][1]:shape[0]*1j,limit_list[1][0]:limit_list[1][1]:shape[1]*1j,limit_list[2][0]:limit_list[2][1]:shape[2]*1j,limit_list[3][0]:limit_list[3][1]:shape[3]*1j]
        elif(len(shape)==5):
            return np.mgrid[limit_list[0][0]:limit_list[0][1]:shape[0]*1j,limit_list[1][0]:limit_list[1][1]:shape[1]*1j,limit_list[2][0]:limit_list[2][1]:shape[2]*1j,limit_list[3][0]:limit_list[3][1]:shape[3]*1j,limit_list[4][0]:limit_list[4][1]:shape[4]*1j]
        elif(len(shape)==6):
            return np.mgrid[limit_list[0][0]:limit_list[0][1]:shape[0]*1j,limit_list[1][0]:limit_list[1][1]:shape[1]*1j,limit_list[2][0]:limit_list[2][1]:shape[2]*1j,limit_list[3][0]:limit_list[3][1]:shape[3]*1j,limit_list[4][0]:limit_list[4][1]:shape[4]*1j,limit_list[5][0]:limit_list[5][1]:shape[5]*1j]
        elif(len(shape)==7):
            return np.mgrid[limit_list[0][0]:limit_list[0][1]:shape[0]*1j,limit_list[1][0]:limit_list[1][1]:shape[1]*1j,limit_list[2][0]:limit_list[2][1]:shape[2]*1j,limit_list[3][0]:limit_list[3][1]:shape[3]*1j,limit_list[4][0]:limit_list[4][1]:shape[4]*1j,limit_list[5][0]:limit_list[5][1]:shape[5]*1j,limit_list[6][0]:limit_list[6][1]:shape[6]*1j]
        elif(len(shape)==8):
            return np.mgrid[limit_list[0][0]:limit_list[0][1]:shape[0]*1j,limit_list[1][0]:limit_list[1][1]:shape[1]*1j,limit_list[2][0]:limit_list[2][1]:shape[2]*1j,limit_list[3][0]:limit_list[3][1]:shape[3]*1j,limit_list[4][0]:limit_list[4][1]:shape[4]*1j,limit_list[5][0]:limit_list[5][1]:shape[5]*1j,limit_list[6][0]:limit_list[6][1]:shape[6]*1j,limit_list[7][0]:limit_list[7][1]:shape[7]*1j]

def grab_args_lite(args):
    shape=args[0].shape
    limit_list=[]
    for i in range(len(shape)):
        limit_list.append([args[i].min(),args[i].max()])
    return limit_list,shape


        
index_roundoff_compensate=2e-14
def log_array_01N(array_element_01N,N,k_init=0.9):
    temp=(array_element_01N[2]-array_element_01N[0])/(array_element_01N[1]-array_element_01N[0])
    def f(k):
        return np.exp(k*N)-1-temp*(np.exp(k)-1)
    if(temp>N):
        sol=opt.root(f,k_init)
        if(sol.success):
            k=sol.x[0]
            a=(array_element_01N[1]-array_element_01N[0])/(np.exp(k)-1)
            return a,k,array_element_01N[0]+a*(np.exp(np.linspace(0,N*k,N+1))-1)
        else:
            print('toolbox error! Try initialize k_init better.')
    else:
        print('toolbox error! Choose a smaller N.')
def log_array(array_lim,delta_factor,N): #delta factor is (y_N-y_{N-1}/(y1-y0)
    k=np.log(delta_factor)/(N-1)
    a=(array_lim[1]-array_lim[0])/(np.exp(k*N)-1)
    return a,k,array_lim[0]+a*(np.exp(np.linspace(0,N*k,N+1))-1)
def log_array_extend(array_lim_min,a,k,N): #delta factor is (y_N-y_{N-1}/(y1-y0)
    return array_lim_min+a*(np.exp(np.linspace(0,N*k,N+1))-1)
def log_index(array_i,array_lim_min,a,k,N,float_or_int='int'):
    array_i=array_i-index_roundoff_compensate
    #print(array_i,array_lim,a,k,-(-np.log(np.where(array_i<array_lim.min(),0,(array_i-array_lim.min())/a)+1)/k).astype('int'))
    return N-1-(N-np.log(np.where(array_i<array_lim_min,0,(array_i-array_lim_min)/a)+1)/k).astype(float_or_int)

import scipy.optimize as opt
def log_array_centered(array_lim,delta_factor,N1_N2):
    N1,N2=N1_N2
    a,k,right_side=log_array(array_lim[1:3],delta_factor,N2)
    def eq(k_,a,k):
        return a*k*(np.exp(k_*N1)-1)-(array_lim[1]-array_lim[0])*k_
    if(right_side[1]-right_side[0]>(array_lim[1]-array_lim[0])/N1):
        left_side=np.linspace(array_lim[0],array_lim[1],N1+1)[:-1]
    else:
        k_=opt.newton(eq,k,args=(a,k))
        a_=a*k/k_
        left_side = log_array_extend(array_lim[1],-a_,k_,N1)[::-1][:-1]
    return np.concatenate((left_side,right_side))

def log_array_centered_smooth(array_lim,N1_N2,log_delta_factor_init=-4):
    def eq(log_delta_factor,array_lim,N1_N2):
        delta_factor=np.exp(log_delta_factor)
        result=log_array_centered(array_lim,delta_factor,N1_N2)
        return result[N1_N2[0]]**2/(result[N1_N2[0]+1]*result[N1_N2[0]-1])-1
    log_delta_factor=opt.newton(eq,log_delta_factor_init,args=(array_lim,N1_N2))
    delta_factor=np.exp(log_delta_factor)
    return log_array_centered(array_lim,delta_factor,N1_N2)

def log_linear_array(array_lim,N1_N2):
    N1,N2=N1_N2
    para=np.zeros(N1+1)
    para[0]=1+N2
    para[1]=-N2
    para[-1]=-array_lim[1]/array_lim[0]
    roots=np.roots(para)[0]
    if(np.abs(roots.imag)<1e-6):
        log_array=array_lim[0]*roots.real**np.linspace(0,N1,N1+1)
        linear_array=np.linspace(log_array[-1],array_lim[1],N2+1)
        return np.concatenate((log_array,linear_array[1:]))

def parameter_f_fit(f_fit,dataX,dataY,dataX_range=[-float("inf"),float("inf")],dataY_range=[-float("inf"),float("inf")]):
    logic=np.logical_and(np.logical_and(dataX>dataX_range[0],dataX<dataX_range[1]),np.logical_and(dataY>dataY_range[0],dataY<dataY_range[1]))
    return opt.curve_fit(f_fit,dataX[logic],dataY[logic])[0]

import matplotlib.pyplot as plt
def plot_MR_causality(axes,R_lim,M_lim,shift=0.,text_fontsize=15,color='lime',ratio=4.170756573956214,text='causality'):#ratio=5.45416088816032 for cs2=1/3
    axes.set_xlim(R_lim[0],R_lim[1])
    axes.set_ylim(M_lim[0],M_lim[1])
    fill_R=np.array([R_lim[0],M_lim[1]*ratio])
    fill_M=np.array([R_lim[0]/ratio,M_lim[1]])
    axes.fill_between(fill_R,fill_M,np.array([M_lim[1],M_lim[1]]),color=color,alpha=0.5)
    angle = np.arctan(1/ratio)*180/np.pi
    position=np.array([fill_R.sum()+shift*np.diff(fill_R),fill_M.sum()+shift*np.diff(fill_M)]).transpose()[0]/2
    trans_angle = plt.gca().transData.transform_angles(np.array((angle,)),
                                                       position.reshape((1, 2)))[0]
    axes.text(position[0],position[1],text,rotation=trans_angle,fontsize=text_fontsize, rotation_mode='anchor')
# =============================================================================
# compact_eos=eos_class.EOS_CSS([150,0,1,1])
# mr=MassRadius(Maxmass(1e-10,1e-8,compact_eos)[1],1e-10,1e-8,'MR',compact_eos)
# mr[1]/(1000*mr[0]) #=4.170756573956214 for cs2=1, 5.45416088816032 for cs2=1/3
# =============================================================================
