#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 15:50:18 2023

@author: smith
"""
#%%
import os
os.chdir('/d1/software/ClearMap_Python3')
import numpy as np
from matplotlib import pyplot as mplt
import matplotlib
matplotlib.rcParams.update({'text.usetex': False, 'font.family': 'stixgeneral', 'mathtext.fontset': 'stix', 'pdf.fonttype':42, 'ps.fonttype':42})

import ClearMap.IO as io
from ClearMap.Analysis.Statistics import thresholdPoints


#%%
sampleName = 'PM1_AC'
mouse=sampleName.split('_')[0]
signalChannel='cfos'
testing=True

#execfile('/d2/studies/CM2/SmartSPIM/BH_Npas4_TRAP/' + sampleName + '_Stitched/clearMap_Ilastik_parameters_'+sampleName+'.py')
paramFile='/d2/studies/SmartSPIM/PM1_AC_Overlay_cfos/clearMap_parameters_PM1_AC_Overlay_npas.py'
BaseDirectory = os.path.dirname(paramFile)
#exec(open(paramFile).read())

#%%
backgroundPath='/d2/studies/SmartSPIM/PM1_AC_Overlay_npas/MAX_PM1_AC_Npas4_Substack_HPC_Adj.tif'

x1=1900
y1=1600
width=300
height=300

#%% Using the raw data & filtering:
points, intensities = io.readPoints(ImageProcessingParameter["sink"]);

print("Unfiltered # of points: " + str(points.shape[0]))
minSize=20
maxSize=1500
print("Filtering out " + str(intensities[intensities[:,1]<minSize].shape[0]) + " points smaller than " + str(minSize) + " pixels")
print("Filtering out " + str(intensities[intensities[:,1]>maxSize].shape[0]) + " points larger than " + str(maxSize) + " pixels")
points, intensities = thresholdPoints(points, intensities, threshold = (minSize, maxSize), row = (1,1));
io.writePoints(FilteredCellsFile, (points, intensities));

#%% Transform points, save cells alone
suffix='PM1_AC_npas_20-1500'
imageType='.pdf'

arr=points[:,:2]
c1 = arr[:,0]-x1
c2=arr[:,1]-y1
c3 = c2-height
c3 = np.abs(c3)
cat = np.stack([c1,c3],axis=1)
#fig = mplt.figure(figsize=(25,25))
fig, ax = mplt.subplots()
ax.set_xlim(0,width)
ax.set_ylim(0,height)
ax.scatter(cat[:,0],cat[:,1], color='red', s=5, marker='.', linewidths=0)
ax.set_aspect(abs(width)/abs(height))
mplt.savefig(os.path.join(BaseDirectory, 'CellCenters_'+suffix+imageType), dpi=300, bbox_inches='tight')#

#%% Overlay on background image (I generate the image in ImageJ)
suffix='PM1_AC_cfos_20-1500'
imageType='.pdf'

mplt.box(False)
bg = mplt.imread(backgroundPath)
#mplt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
#mplt.tick_params(axis='y', which='both', bottom=False, top=False, labelbottom=False)
fig, ax = mplt.subplots()
img = ax.imshow(bg, zorder=0, extent=[0,height,0,width], cmap='binary')
#mplt.savefig(os.path.join(BaseDirectory, 'DataSubset_bg.tif'), dpi=300, bbox_inches='tight')
ax.scatter(cat[:,0], cat[:,1], color='red', s=10, marker='.', alpha=0.8, linewidths=0)
mplt.savefig(os.path.join(BaseDirectory, 'CellMarkerOverlay_'+suffix+imageType), dpi=300, bbox_inches='tight', transparent=imageType=='.pdf')
mplt.close('all')

#%%
def overlayPoints(sampleName, BaseDirectory, img, x1=350, y1=650, width=300, height=300,
                  use_filtered=False, minSize=20, maxSize=2000, imageType='.pdf',
                  overwrite_points=False, size=5, color='red', suffix='', dpi=300):
  #  exec(open(paramFile).read())
    SpotDetectionParameter = {
    "sink"   : (os.path.join(BaseDirectory, sampleName+'_cells-allpoints.npy'),  os.path.join(BaseDirectory,  sampleName+'_intensities-allpoints.npy')),
    };
    FilteredCellsFile = (os.path.join(BaseDirectory, sampleName+'_cells.npy'), os.path.join(BaseDirectory, sampleName+'_intensities.npy'));

    #Read points from ClearMap analysis
    if not use_filtered:
        points, intensities = io.readPoints(SpotDetectionParameter["sink"]);
        print("Unfiltered # of points: " + str(points.shape[0]))
        minSize=minSize
        maxSize=maxSize
        print("Filtering out " + str(intensities[intensities[:,1]<minSize].shape[0]) + " points smaller than " + str(minSize) + " pixels")
        print("Filtering out " + str(intensities[intensities[:,1]>maxSize].shape[0]) + " points larger than " + str(maxSize) + " pixels")
        points, intensities = thresholdPoints(points, intensities, threshold = (minSize, maxSize), row = (1,1));
        if overwrite_points:
            io.writePoints(FilteredCellsFile, (points, intensities));
    elif use_filtered:
        points, intensities = io.readPoints(FilteredCellsFile)
        
    #Transform points to 2D array
    arr=points[:,:2]
    c1 = arr[:,0]-x1
    c2 = arr[:,1]-y1
    c3 = np.abs(c2-height)
    cat = np.stack([c1,c3],axis=1)
    
    #Plot figure of only points:
    fig, ax = mplt.subplots()
    ax.set_xlim(0,width)
    ax.set_ylim(0,height)
    ax.scatter(cat[:,0],cat[:,1], color=color, s=size, marker='.', linewidths=0)
    ax.set_aspect(abs(width)/abs(height))
    mplt.savefig(os.path.join(BaseDirectory, 'CellCenters_'+suffix+imageType), dpi=dpi, bbox_inches='tight')#
    
    #Plot points overlayed on image
    mplt.box(False)
    bg = mplt.imread(img)
    fig, ax = mplt.subplots()
    ax.imshow(bg, zorder=0, extent=[0,height,0,width], cmap='binary')
    ax.scatter(cat[:,0], cat[:,1], color='red', s=10, marker='.', alpha=0.8, linewidths=0)
    mplt.savefig(os.path.join(BaseDirectory, 'CellMarkerOverlay_'+suffix+imageType), dpi=300, bbox_inches='tight', transparent=imageType=='.pdf')
    mplt.close('all')
    


