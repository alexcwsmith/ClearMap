# -*- coding: utf-8 -*-
"""
Template to run the processing pipeline
"""
#load the parameters:
sampleName='LSM1R'
testing=False
execfile('/d2/studies/CM2/SmartSPIM/ReichelLab/'+sampleName+'_Stitched_stacks/clearMap_parameters_'+sampleName+'.py')

#resampling operations:
#######################
#resampling for the correction of stage movements during the acquisition between channels:
resampleData(**CorrectionResamplingParametersignal);
resampleData(**CorrectionResamplingParameterAutoFluo);

#Downsampling for alignment to the Atlas:
resampleData(**RegistrationResamplingParameter);


#Alignment operations:
######################
#correction between channels:
resultDirectory  = alignData(**CorrectionAlignmentParameter);

#alignment to the Atlas:
resultDirectory  = alignData(**RegistrationAlignmentParameter);

#execfile('/d2/studies/CM2/SmartSPIM/ReichelLab/'+sampleName+'_Stitched_stacks/clearMap_parameters_'+sampleName+'.py')

#Cell detection:
################
detectCells(**ImageProcessingParameter);

#Filtering of the detected peaks:
#################################
#Loading the results:
points, intensities = io.readPoints(ImageProcessingParameter["sink"]);

topPerc = numpy.percentile(intensities[:,3], 99.9)
fig = mplt.figure()
vox = intensities[intensities[:,3]<topPerc][:,3]
mplt.hist(vox, bins=5)
#mplt.show()
fig.savefig(os.path.join(BaseDirectory, sampleName+'_voxelSizeHistogram_'+region+'.png'))


#Thresholding: the threshold parameter is either intensity or size in voxel, depending on the chosen "row"
#row = (0,0) : peak intensity from the raw data
#row = (1,1) : peak intensity from the DoG filtered data
#row = (2,2) : peak intensity from the background subtracted data
#row = (3,3) : voxel size from the watershed
print("Unfiltered # of points: " + str(points.shape[0]))
minSize=12
maxSize=200
print("Filtering out " + str(intensities[intensities[:,3]<minSize].shape[0]) + " points smaller than " + str(minSize) + " pixels")
print("Filtering out " + str(intensities[intensities[:,3]>maxSize].shape[0]) + " points larger than " + str(maxSize) + "pixels")
points, intensities = thresholdPoints(points, intensities, threshold = (minSize, maxSize), row = (3,3));
io.writePoints(FilteredCellsFile, (points, intensities));


## Check Cell detection (For the testing phase only, remove when running on the full size dataset)
#########################
if testing:
    import ClearMap.Visualization.Plot as plt;
    pointSource= os.path.join(BaseDirectory, FilteredCellsFile[0]);
    data = plt.overlayPoints(signalFile, pointSource, pointColor = None, **signalFileRange);
    io.writeData(os.path.join(BaseDirectory, sampleName+'cells_check_'+region+'_'+str(minSize)+'-'+str(maxSize)+'px.tif'), data);


# Transform point coordinates
#############################
points = io.readPoints(CorrectionResamplingPointsParameter["pointSource"]);
points = resamplePoints(**CorrectionResamplingPointsParameter);
points = transformPoints(points, transformDirectory = CorrectionAlignmentParameter["resultDirectory"], indices = False, resultDirectory = None);
CorrectionResamplingPointsInverseParameter["pointSource"] = points;
points = resamplePointsInverse(**CorrectionResamplingPointsInverseParameter);
RegistrationResamplingPointParameter["pointSource"] = points;
points = resamplePoints(**RegistrationResamplingPointParameter);
points = transformPoints(points, transformDirectory = RegistrationAlignmentParameter["resultDirectory"], indices = False, resultDirectory = None);
io.writePoints(TransformedCellsFile, points);




# Heat map generation
#####################
points = io.readPoints(TransformedCellsFile)
intensities = io.readPoints(FilteredCellsFile[1])

#Without weigths:
vox = voxelize(points, AtlasFile, **voxelizeParameter);
if not isinstance(vox, basestring):
  io.writeData(os.path.join(BaseDirectory, sampleName+'_cells_heatmap_vox15.tif'), vox.astype('int32'));
#

#With weigths from the intensity file (here raw intensity):
voxelizeParameter["weights"] = intensities[:,0].astype(float);
vox = voxelize(points, AtlasFile, **voxelizeParameter);
if not isinstance(vox, basestring):
  io.writeData(os.path.join(BaseDirectory, sampleName+'_cells_heatmap_weighted_vox15.tif'), vox.astype('int32'));
#

###Counts analysis
regs = pd.read_csv(regionIndex, index_col=0)

#Without weigths (pure cell number):
ids, counts = countPointsInRegions(points, labeledImage = AnnotationFile, intensities = None, collapse=None)
table = numpy.zeros(ids.shape, dtype=[('id','int64'),('counts','f8'),('name', 'a256')])
table["id"] = ids
table["counts"] = counts
table["name"] = labelToName(ids)
df = pd.DataFrame(table)
df.set_index('id', inplace=True)
df.sort_values(by='counts', ascending=False, inplace=True)
df.to_csv(os.path.join(BaseDirectory, sampleName + '_Annotated_counts.csv'))

ids, counts = countPointsInRegions(points, labeledImage = AnnotationFile, intensities = None, collapse=True)
table = numpy.zeros(ids.shape, dtype=[('id','int64'),('counts','f8'),('name', 'a256')])
table["id"] = ids
table["counts"] = counts
table["name"] = labelToName(ids)
df = pd.DataFrame(table)
df.set_index('id', inplace=True)
df.sort_values(by='counts', ascending=False, inplace=True)
df.to_csv(os.path.join(BaseDirectory, sampleName + '_Annotated_counts_collapse.csv'))

#####################
#####################
#####################
#####################
#####################
#####################
