#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 20:56:31 2023

@author: smith
"""

import ClearMap.Analysis.Label as lbl
import numpy as np
import ClearMap.IO.IO as io
#Read the annotation file & make an array with all unique label IDs:
label = io.readData('AnnotationFile.ome.tif').astype('int32')
labelids = np.unique(label)
#Loop through all IDs and find out if they are a child of the desired label & mask out if not:
outside = np.zeros(label.shape, dtype = bool);
for l in labelids:
    if not (lbl.labelAtLevel(l, 5) == 1089):
       outside = np.logical_or(outside, label == l);
#Read heatmap, apply mask, write result image:
heatmap = io.readData('cells_heatmap.tif')
heatmap[outside] = 0
io.writeData('Example_HPF_isolated.tif', heatmap)


