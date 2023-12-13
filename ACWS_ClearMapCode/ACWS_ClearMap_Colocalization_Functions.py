#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 11:18:57 2023

@author: smith
"""
#%%
import os
os.chdir('/d1/software/ClearMap_Python3')

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

import ClearMap.IO as io
from skimage import io as skio
from skimage.util import img_as_float, img_as_uint
#%%
sampleName = 'MJF-CS1NP'

BaseDirectory='/d2/studies/SmartSPIM/MJF_CS_Cohort1/Coloc_batch'

cfos = np.load('/d2/studies/SmartSPIM/MJF_CS_Cohort1/'+sampleName+'_Stitched'+'/'+sampleName+'_cells_cfos.npy')
rfp = np.load('/d2/studies/SmartSPIM/MJF_CS_Cohort1/'+sampleName+'_Stitched'+'/'+sampleName+'_cells_RFP.npy')

regionName='AGI'
x1=300
y1=200
z1=600
width=1700
height=1700
depth=500
results={}
results.update({'cfos':cfos_filt.shape[0]})
results.update({'rfp':rfp_filt.shape[0]})
            
#%% 
def find_coloc_points(array1, array2, threshold=1.5):
    """
    Given two arrays of points, i.e. the -cells.npy files from ClearMap, finds places where both
    arrays contain a point within a distance less than threshold.
    
    Paramaters
    ----------
    array1 : np.ndarray or str
        First array of points, or path to cells.npy file to load.
    array2 : np.ndarray or str
        Second array of points, or path to cells.npy file to load.
    threshold : float (optional)
        Distance threshold for colocalization. The default is 1.5
        
    Returns
    -------
    coloc : np.ndarray
        Masked array of colocalized points
    
    """
    if isinstance(array1, str):
        array1 = np.load(array1)
    if isinstance(array2, str):
        array2 = np.load(array2)

    dists = cdist(array1, array2)
    mask = dists.min(axis=1) < threshold
    coloc = array1[mask]
    nn = dists.min(axis=1)[mask]
    idx = np.where(mask)[0]

    return coloc, nn, idx

#%%
def filter_points(points, x1=1900, y1=1600, z1=500, width=300, height=300, zdepth=50):
    """
    For filtering a subset of points to plot & evaluate results.
    
    Parameters
    ----------
    coloc_points : np.array
        Array of points points to filter
    x1 : int
        Left edge on X axis
    y1 : int
        Top edge on Y axis
    z1 : int
        First z plane
    width : int
        Width of window to mask
    Height : int
        Height of window to mask
    zdepth : int
        How many z-planes to include
    """
    mask = (points[:, 0] >= x1) & (points[:, 0] < x1 + width) & \
           (points[:, 1] >= y1) & (points[:, 1] < y1 + height) & \
           (points[:, 2] >= z1) & (points[:, 2] < z1 + zdepth)
    filtered = points[mask]
    return filtered

#%%Filter points (can do this before or after running coloc):
cfos_filt = filter_points(cfos, x1, y1, z1, width, height, depth)
rfp_filt = filter_points(rfp, x1, y1, z1, width, height, depth)

#%% Find colocalized points and filter points for a given region
coloc, nn, idx = find_coloc_points(cfos_filt, rfp_filt, threshold=2) #coloc=coordinate of point in array1, nn=nearest neighbor in array2, idx=index of colocalized points in array1
#use the below line if you did not filter points prior to doing coloc, and you want to:

np.save(os.path.join(BaseDirectory, sampleName+'_cells-coloc.npy'), coloc)    

#%%Make and save csv of numerical results
results.update({'cfos_filt':cfos_filt.shape[0]})
results.update({'rfp_filt':rfp_filt.shape[0]})
            

results.update({'coloc':coloc.shape[0]})
results.update({'% total coloc':coloc.shape[0]/sum([cfos_filt.shape[0],rfp_filt.shape[0]])})

df = pd.DataFrame(results, index=['counts']).T
df.to_csv(os.path.join(BaseDirectory, 'Colocalized_Counts_Thresh3.csv'))
#%% Make 2D array
arr=filt[:,:2]
c1 = arr[:,0]-x1
c2=arr[:,1]-y1
c3 = np.abs(c2-height)
cat = np.stack([c1,c3],axis=1)

#%%
def make_transform_2d_array(points, x1=1900, y1=1600, width=300, height=300):
    """
    Flattens a 3D array of points into 2d (i.e. removes Z coordinates)

    Parameters
    ----------
    points : np.array
        Array of points as (X,Y,Z) coordinates
    x1 : int, optional
        Left edge in x axis. The default is 1900.
    y1 : int, optional
        Top edge in Y axis. The default is 1600.
    width : int, optional
        Width of image to view. The default is 300.
    height : int, optional
        Height of image to view. The default is 300.

    Returns
    -------
    cat : np.array
        2-dimensional array of points within bounding box, ready to overlay on image.

    """
    arr = points[:,:2]
    c1 = arr[:,0]-x1
    c2 = arr[:,1]-y1
    c3 = np.abs(c2-height)
    cat = np.stack([c1,c3],axis=1)
    return cat
  

#%% Make 3D array
def img_from_points(points, x1, y1, z1, width=300, height=300, depth=50, name='cFos_Points', imageType='.tif', dtype=np.uint16):
    if type(points)=='str':
        points=np.load(points)
    x_arr=points[:,0]-x1
    y_arr=points[:,1]-y1
    z_arr=points[:,2]-z1
    #cat = np.stack([c1,c2,c3], axis=2)
    stack = np.column_stack([z_arr, y_arr, x_arr])
    newImg = np.zeros(shape=(depth,width,height))
    newImg[stack[:,0], stack[:,1], stack[:,2]] = 1
    if dtype==np.uint16:
        skio.imsave(os.path.join(BaseDirectory, name+imageType), img_as_uint(newImg))
    elif dtype==np.float32:
        skio.imsave(os.path.join(BaseDirectory, name+imageType), img_as_float(newImg))

#%%
flat = make_transform_2d_array(coloc, x1=600, y1=400, width=500, height=500)

#%%
img_from_points(coloc, x1, y1, z1, width, height, depth, name='Coloc_points_CS1NP_600-400-800_XYZ_RFP_minSize8_Thresh3', imageType='.tif')

#%%Plot colocalized cell centers only
suffix='17OCT_1NP_Coloc_cFos_RFP_aIC_466pts'
imageType='.png'

from matplotlib import pyplot as mplt
fig, ax = mplt.subplots()
ax.set_xlim(0,width)
ax.set_ylim(0,height)
ax.scatter(cat[:,0],cat[:,1], color='red', s=5, marker='.', linewidths=0)
ax.set_aspect(abs(width)/abs(height))
mplt.savefig(os.path.join(BaseDirectory, 'CellCenters_'+suffix+imageType),
             dpi=300, bbox_inches='tight', transparent=imageType=='.pdf')#


#%%Make cell centers with no frame
suffix='NoFrame'
imageType='.png'

fig, ax = mplt.subplots()
mplt.box(False)
ax.scatter(cat[:,0],cat[:,1], color='red', s=5, marker='.', linewidths=0)
ax.axis('off')
ax.set_aspect(abs(width)/abs(height))
mplt.savefig(os.path.join(BaseDirectory, 'CellCenters_'+suffix+imageType),
             dpi=300, bbox_inches='tight', transparent=imageType=='.pdf')#
mplt.close('all')

#%% Overlay on background image (I generate the image in ImageJ)
suffix='CS1NP_Coloc_OverRFP'
imageType='.png'
backgroundPath='/d2/studies/SmartSPIM/MJF_CS_Cohort1/Coloc_MJF-CS1NP/MAX_MJF-CS1NP-Substack_X600_Y400_Z800_RFP_LUT.tif'

mplt.box(False)
bg = mplt.imread(backgroundPath)
#mplt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
#mplt.tick_params(axis='y', which='both', bottom=False, top=False, labelbottom=False)
fig, ax = mplt.subplots()
img = ax.imshow(bg, zorder=0, extent=[0,height,0,width], cmap='binary')
#mplt.savefig(os.path.join(BaseDirectory, 'DataSubset_bg.tif'), dpi=300, bbox_inches='tight')
ax.scatter(cat[:,0], cat[:,1], color='red', s=2, marker='.', alpha=0.8, linewidths=0)
mplt.savefig(os.path.join(BaseDirectory, 'CellMarkerOverlay_'+suffix+imageType), dpi=300, bbox_inches='tight', transparent=imageType=='.pdf')
mplt.close('all')

#%% Overlay on background image 2
suffix='PM1_Coloc_OverNpas'
imageType='.png'
backgroundPath='/d2/studies/SmartSPIM/PM1_AC_Overlay_npas/MAX_PM1_AC_Npas4_Substack_HPC_Adj.tif'

mplt.box(False)
bg = mplt.imread(backgroundPath)
#mplt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
#mplt.tick_params(axis='y', which='both', bottom=False, top=False, labelbottom=False)
fig, ax = mplt.subplots()
img = ax.imshow(bg, zorder=0, extent=[0,height,0,width], cmap='binary')
#mplt.savefig(os.path.join(BaseDirectory, 'DataSubset_bg.tif'), dpi=300, bbox_inches='tight')
ax.scatter(cat[:,0], cat[:,1], color='red', s=2, marker='.', alpha=0.8, linewidths=0)
mplt.savefig(os.path.join(BaseDirectory, 'CellMarkerOverlay_'+suffix+imageType), dpi=300, bbox_inches='tight', transparent=imageType=='.pdf')
mplt.close('all')

#%% Turn coordinates into image

def img_from_coords(coords, name='Coloc_Points', shape=(300,300), plugin='skimage'):
    arr = np.zeros(shape, dtype=np.float32)
    if coords.shape[1]==3:
        arr[coords[:,0], coords[:,1], coords[:,2]] = 1
        transp = arr.transpose(2,1,0)
        flip = np.flip(transp, 1)
    elif coords.shape[1]==2:
        arr[coords[:,0], coords[:,1]] = 1
        transp = arr.transpose(1,0)
        flip = np.flip(transp, 0)
    else:
        raise IndexError("Input coordinates must either be either 3D (X,Y,Z) or 2D (X,Y) coordinates.\n"
                         "Found array with shape " + str(coords.shape))
    if plugin=='skimage':
        skio.imsave(os.path.join(BaseDirectory, name+'.tif'), img_as_float(flip))
    return arr




