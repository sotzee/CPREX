import numpy as np
from scipy import interpolate
from matplotlib_rc import *

from collections import Counter
def get_density_2D(points_array,bin_num):
    points_array_x=points_array[0]
    points_array_y=points_array[1]
    bin_size_x=(points_array_x.max()-points_array_x.min())/(2*bin_num[0])
    bin_size_y=(points_array_y.max()-points_array_y.min())/(2*bin_num[1])
    x_lim = np.array([points_array_x.min(), points_array_x.max()])+0.5*bin_size_x
    y_lim = np.array([points_array_y.min(), points_array_y.max()])+0.5*bin_size_y
    xx, yy = np.mgrid[x_lim[0]:x_lim[1]:(bin_num[0]*1j), y_lim[0]:y_lim[1]:(bin_num[1]*1j)]
    
    points_array_x_int=((1./bin_size_x)*(points_array_x-points_array_x.min())).astype(int)
    points_array_y_int=((1./bin_size_y)*(points_array_y-points_array_y.min())).astype(int)
    points_array_x_int[points_array_x_int==bin_num[0]]=bin_num[0]-1
    points_array_y_int[points_array_y_int==bin_num[1]]=bin_num[1]-1
    points_array_int=np.array([points_array_x_int,points_array_y_int]).transpose()
    points_array_str=[str(a[0])+','+str(a[1]) for a in points_array_int]
    points_array_str_counts=Counter(points_array_str)
    counting_array=np.zeros(bin_num)
    for i in range(bin_num[0]):
        for j in range(bin_num[1]):
            counting_array[i,j]=points_array_str_counts['%d,%d'%(i,j)]
    return xx, yy, counting_array

from scipy.stats import gaussian_kde
def get_kde_1D(points_array_x,bin_num,weights=None,x_min=np.infty,x_max=-np.infty,bw_method=None):
    x_lim = np.array([min(points_array_x.min(),x_min), max(points_array_x.max(),x_max)])
    positions = np.linspace(x_lim[0],x_lim[1],bin_num)
    kernel = gaussian_kde(points_array_x,weights=weights,bw_method=bw_method)
    kde_array = kernel(positions)
    return positions, kde_array

def get_kde_2D(points_array,bin_num,weights=None,x_min=np.infty,x_max=-np.infty,y_min=np.infty,y_max=-np.infty,kde_fun=False,bw_method=None):
    points_array_x=points_array[0]
    points_array_y=points_array[1]
    bin_size_x=(points_array_x.max()-points_array_x.min())/(2*bin_num[0])
    bin_size_y=(points_array_y.max()-points_array_y.min())/(2*bin_num[1])
    x_lim = np.array([min(points_array_x.min(),x_min), max(points_array_x.max(),x_max)])+0.5*bin_size_x
    y_lim = np.array([min(points_array_y.min(),y_min), max(points_array_y.max(),y_max)])+0.5*bin_size_y
    xx, yy = np.mgrid[x_lim[0]:x_lim[1]:(bin_num[0]*1j), y_lim[0]:y_lim[1]:(bin_num[1]*1j)]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([points_array_x, points_array_y])
    kernel = gaussian_kde(values,weights=weights,bw_method=bw_method)
    kde_array = np.reshape(kernel(positions).T, xx.shape)
    if(kde_fun):
        return xx, yy, kde_array, kernel
    else:
        return xx, yy, kde_array
    
import scipy.optimize as opt
def plot_density_1D(x,density_array,percentile,color_list,ax,marginal_axis='x',unit='',legend_loc=0,figsize_norm=1,n=100,label_fmt='%.3f - %.3f'):
    density_array=density_array/density_array.sum()
    density_array_max=density_array.max()
    t = np.linspace(0, density_array_max, n)
    integral = ((density_array >= t[:, None]) * density_array).sum(axis=(1))
    f = interpolate.interp1d(integral, t)
    t_contours = f(np.array(percentile))
# =============================================================================
#     f = interpolate.interp1d(x,density_array)
#     opt.newton(f,t_contours,args=[])
#     print f(t_contours)
# =============================================================================
    x_contours = []
    density_countours = []
    for index_list_flag in density_array>t_contours[:,None]:
        index_list=np.where(index_list_flag)[0]
        x_contours.append(x[[index_list.min(),index_list.max()]])
        density_countours.append(density_array[[index_list.min(),index_list.max()]])
    if(marginal_axis=='x'):
        ax.plot(x,density_array,linewidth=5*figsize_norm)
        for i in range(len(percentile)):
            ax.plot([x_contours[i][0],x_contours[i][0]],[0,density_countours[i][0]],'--',color=color_list[i],linewidth=5*figsize_norm)
            ax.plot([x_contours[i][1],x_contours[i][1]],[0,density_countours[i][1]],'--',color=color_list[i],linewidth=5*figsize_norm,label=label_fmt%(x_contours[i][0],x_contours[i][1])+unit)
        ax.tick_params(labelbottom=False, labelleft=False)
        ax.legend(fontsize=25*figsize_norm,frameon=False,loc=legend_loc)
    elif(marginal_axis=='y'):
        ax.plot(density_array,x,linewidth=5*figsize_norm)
        for i in range(len(percentile)):
            ax.plot([0,density_countours[i][0]],[x_contours[i][0],x_contours[i][0]],'--',color=color_list[i],linewidth=5*figsize_norm)
            ax.plot([0,density_countours[i][1]],[x_contours[i][1],x_contours[i][1]],'--',color=color_list[i],linewidth=5*figsize_norm,label=label_fmt%(x_contours[i][0],x_contours[i][1])+unit)
        ax.tick_params(labelbottom=False, labelleft=False)
        ax.legend(fontsize=25*figsize_norm,frameon=False,loc=legend_loc)
    #return x, density_array

import matplotlib.gridspec as gridspec
def plot_density_2D(xx,yy,density_array,percentile,color_list,label_x,label_y,x_unit='',y_unit='',legend_loc=[0,0,0],figsize_norm=1,n=20,inline=True,label_fmt='%.3f - %.3f'):
    density_array = density_array / (density_array.sum())#*4*half_bin_size_x*half_bin_size_y)
    f = plt.figure(figsize=(20*figsize_norm,20*figsize_norm))
    gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[1, 4],wspace=0.05,hspace=0.05)
    xy_density = plt.subplot(gs[2])
    xy_density.tick_params(labelsize=40*figsize_norm)
    t = np.linspace(0, density_array.max(), n)
    integral = ((density_array >= t[:, None, None]) * density_array).sum(axis=(1,2))
    f = interpolate.interp1d(integral, t)
    t_contours = f(np.array(percentile))
    label_contours = np.array(100*np.array(percentile)).astype('str')
    print(label_contours)
    cs = xy_density.contour(xx,yy,density_array, t_contours,linestyles='--',colors=color_list,linewidths=5*figsize_norm)
    cf = xy_density.contourf(xx, yy, density_array,100, cmap='Blues')
    #cf.colorbar()
    peak_density=np.where(density_array==density_array.max())
    xy_density.plot([xx[peak_density]],[yy[peak_density]],'+')
    fmt = {}
    print(cs.levels)
    for i in range(len(cs.levels)):
        fmt[cs.levels[i]] = label_contours[i]+'%'
        cs.collections[i].set_label(label_contours[i]+'%')
    xy_density.clabel(cs, cs.levels, inline=inline, fmt=fmt, fontsize=10*figsize_norm)
    xy_density.set_xlabel(label_x,fontsize=40*figsize_norm)
    xy_density.set_ylabel(label_y,fontsize=40*figsize_norm)
    xy_density.legend(fontsize=40*figsize_norm,frameon=False,loc=legend_loc[0])
    x_marginal = plt.subplot(gs[0],sharex=xy_density)
    plot_density_1D(xx[:,0],density_array.sum(1),percentile,color_list,x_marginal,marginal_axis='x',unit=x_unit,legend_loc=legend_loc[1],figsize_norm=figsize_norm,n=n,label_fmt=label_fmt)
    
    y_marginal = plt.subplot(gs[3],sharey=xy_density)
    plot_density_1D(yy[0],density_array.sum(0),percentile,color_list,y_marginal,marginal_axis='y',unit=y_unit,legend_loc=legend_loc[2],figsize_norm=figsize_norm,n=n,label_fmt=label_fmt)
    
    xy_density.set_xlim(xx[0,0],xx[-1,0])
    xy_density.set_ylim(yy[0,0],yy[0,-1])
    return xy_density