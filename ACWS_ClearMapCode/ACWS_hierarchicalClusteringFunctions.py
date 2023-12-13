#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 16:53:11 2022

@author: smith
"""
#%%
import os
import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as sch
from scipy.spatial import distance
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import zscore
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
    
#%%
def dendrogram(var, name, regions, flavor='scipy', corr_method='pearson', linkage_method='complete',
               optimal_order=False, imageType='.svg', saveDir=os.getcwd(),
               saveDetails=True, plotCorrelations=True, imageSize=(200,200), fontSize=1, dpi=300):
    corr_array = pd.DataFrame(var).T.corr(method=corr_method)
    if flavor=='scanpy':
         corr_condensed = distance.squareform(1-corr_array)
         linkage = sch.linkage(
             corr_condensed, method=linkage_method, optimal_ordering=optimal_order
         )
    else:
        pairwise_distances = sch.distance.pdist(corr_array)
        try:
            linkage = sch.linkage(pairwise_distances, method=linkage_method)
        except ValueError:
            print("NaN or inf values found in " + name + " pairwise distances, correcting...")
            imd = np.nan_to_num(pairwise_distances, nan=1)
            linkage = sch.linkage(imd, method=linkage_method)
     
    dend = sch.dendrogram(linkage, labels=regions['name'].tolist(), no_plot=False, leaf_font_size=2)
    regions_reorder = dend['ivl']
    clusters = dend['leaves_color_list']
    if optimal_order:
        plt.savefig(os.path.join(saveDir, 'SciPyDendrogram_'+name+'_optimalOrdering_'+flavor+imageType), bbox_inches='tight', dpi=300)
    else:
        plt.savefig(os.path.join(saveDir, 'SciPyDendrogram_'+'_'.join([name, flavor, imageType])), transparent=imageType=='.pdf', bbox_inches='tight', dpi=300)
    plt.close()

    if plotCorrelations:
        idx = dend['leaves']
        result = corr_array.iloc[idx, :].T.iloc[idx, :]
       # regions_reorder = regions.iloc[idx,:]

        fig = plt.Figure(figsize=imageSize)
        sns.heatmap(result, cmap=plt.cm.bwr,vmin=-1,vmax=1)
        plt.yticks(np.arange(result.shape[0]), regions_reorder, fontsize=fontSize)
        plt.xticks(np.arange(result.shape[0]), regions_reorder, fontsize=fontSize, rotation='vertical')
        plt.savefig(os.path.join(saveDir, "DendrogramCorrelations_"+name+imageType), bbox_inches='tight', dpi=dpi, transparent=imageType=='.pdf')
        plt.close()
        
    if saveDetails:
        df = pd.DataFrame([regions_reorder, clusters]).T
        df.columns=['region', 'cluster']
        df.to_csv(os.path.join(saveDir, name+'_DendrogramInfo.csv'))

#%%
def clusterCorrelations(var, name, regions, saveDir, flavor='scipy', corr_method='pearson', linkage_method='complete', criterion='distance',
                        n_clusters=None, dist_threshold=None, inplace=False, plot=True, imageType='.png', 
                        imageSize=(200,200), fontSize=1, dpi=300, plotDendrogram=True, save_matrix=True):
    """
    Performs hierarchical clustering on an array, grouping highly correlated variables (regionns) into clusters.
    Optionally plots the rearranged correlation matrix.
    
        
    Parameters
    ----------
    var : np.array
        The array to cluster
    
    name : string
        The name for labeling / saving files.
    
    regions:
        Ordered list of regions to use as index.
        
    inplace: bool, optional
        Whether to perform in place or return a new array.
        
    saveClusters: string or None, optional
        
        
    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix with the columns and rows rearranged
    """
    if criterion=='maxcluster' and n_clusters==None:
        raise ValueError("If using 'maxclust' criterion you must specify n_clusters.")
    corr_array=pd.DataFrame(var).T.corr(method=corr_method)
    if flavor=='scanpy':
        pairwise_distances=sch.distance.squareform(1-corr_array)
    else:
        pairwise_distances = sch.distance.pdist(corr_array)
    try:
        linkage = sch.linkage(pairwise_distances, method=linkage_method)
    except ValueError:
        print("NaN or inf values found in " + name + " pairwise distances, correcting...")
        imd = np.nan_to_num(pairwise_distances, nan=1)
        linkage = sch.linkage(imd, method=linkage_method)
    if dist_threshold==None:
        cluster_distance_threshold = pairwise_distances.max()/2
    else:
        cluster_distance_threshold=dist_threshold
    if criterion=='maxclust':
        labels = sch.fcluster(linkage, n_clusters, criterion='maxclust')
    elif criterion=='distance':
        labels = sch.fcluster(linkage, cluster_distance_threshold, criterion='distance')
    idx = np.argsort(labels)
    c, coph = sch.cophenet(linkage, pairwise_distances)
    print(name + " Cophenetic distance " + str(c))
            
    if not inplace:
        corr_array = corr_array.copy()
        
    if isinstance(corr_array, pd.DataFrame):
        result = corr_array.iloc[idx, :].T.iloc[idx, :]
    else:
        result = corr_array[idx,:][:,idx]
    
    clusters = pd.DataFrame([regions['name'].tolist(), labels]).T
    clusters.columns=['Region', 'cluster']
    clusters.sort_values(by='cluster', inplace=True)
    clusters.to_csv(os.path.join(saveDir, name+'_clusterAssignments.csv'))
    clusters['combined']=clusters['Region']+'-'+clusters['cluster'].astype(str)
    
    if plotDendrogram:
        plt.figure(figsize=(25,10))
        plt.title("Dendrogram")
        plt.xlabel('Region Name')
        plt.ylabel('Distance')
        sch.dendrogram(linkage, leaf_rotation=90., leaf_font_size=8, labels=regions['name'].tolist(), color_threshold=8)
        plt.axhline(y=cluster_distance_threshold, c='k')
        if imageType=='.pdf':
            plt.savefig(os.path.join(saveDir, name+'_ClusterCorr_Dendrogram_CutTree'+str(round(cluster_distance_threshold,2))+imageType), bbox_inches='tight', transparent=True, dpi=dpi)
        else:
            plt.savefig(os.path.join(saveDir, name+'_ClusterCorr_Dendrogram_CutTree'+str(round(cluster_distance_threshold,2))+imageType), bbox_inches='tight', dpi=dpi)
        plt.close('all')

    if plot:
        result['name']=regions['name']
        result.set_index('name', inplace=True)
        
        plt.Figure(figsize=imageSize)
        sns.heatmap(result, cmap=plt.cm.bwr,vmin=-1,vmax=1)
        plt.yticks(np.arange(result.shape[0]), clusters['combined'].tolist(), fontsize=fontSize)
        plt.xticks(np.arange(result.shape[0]), clusters['combined'].tolist(), fontsize=fontSize, rotation='vertical')

        if isinstance(imageType, list):
            for t in imageType:
                try:
                    plt.savefig(os.path.join(saveDir, "HierarchicalClustering_"+name+t), bbox_inches='tight', transparent=True, dpi=dpi)
                except FileNotFoundError:
                    print("Save directory not found. saveDir argument should be a string path to directory, or False. Saving to home directory instead.")
                    plt.savefig( "~/HierarchicalClustering_"+name+t, bbox_inches='tight', transparent=True, dpi=dpi)
                   
        else:
            try:
                plt.savefig(os.path.join(saveDir, "HierarchicalClustering_"+name+imageType), bbox_inches='tight', transparent=True, dpi=dpi)
            except FileNotFoundError:
                print("Save directory not found. saveDir argument should be a string path to directory, or False. Saving to home directory instead.")
                plt.savefig( "~/HierarchicalClustering_"+name+imageType, bbox_inches='tight', dpi=dpi)                
        plt.close('all')
        
    if save_matrix:
        result.columns=result.index
        result.to_csv(os.path.join(saveDir, name+'_ClusteredCorrelationMatrix.csv'))
        
    return result

#%%Reorder correlation matrix to match clustered data
def reorderCorrs(data, reference, name, saveDir, suffix=None, save_matrix=False,
                 plot=True, imageSize=(200,200), imageType='.pdf', dpi=300, fontSize=1):
    df = pd.read_csv(data, index_col=0)
    ref = pd.read_csv(reference, index_col=0)
    order = ref.index.tolist()
    names = ref['Region'].tolist()
    result = df.iloc[order,:].T.iloc[order,:]
    result = result.astype('float32')
    result.index=names
    result.columns=names
    if suffix:
        name='_'.join([name, suffix])
    
    if save_matrix:
        result.to_csv(os.path.join(saveDir, name+'.csv'))
    
    if plot:
        plt.Figure(figsize=imageSize)
        sns.heatmap(result, cmap=plt.cm.bwr,vmin=-1,vmax=1)
        plt.yticks(np.arange(result.shape[0]), result.index.tolist(), fontsize=fontSize)
        plt.xticks(np.arange(result.shape[0]), result.index.tolist(), fontsize=fontSize, rotation='vertical')
    
        
    
        if isinstance(imageType, list):
            for t in imageType:
                try:
                    plt.savefig(os.path.join(saveDir, "HierarchicalClustering_"+name+t), bbox_inches='tight', transparent=True, dpi=dpi)
                except FileNotFoundError:
                    print("Save directory not found. saveDir argument should be a string path to directory, or False. Saving to home directory instead.")
                    plt.savefig( "~/HierarchicalClustering_"+name+t, bbox_inches='tight', transparent=True, dpi=dpi)
                   
        else:
            try:
                plt.savefig(os.path.join(saveDir, "HierarchicalClustering_"+name+imageType), bbox_inches='tight', transparent=True, dpi=dpi)
            except FileNotFoundError:
                print("Save directory not found. saveDir argument should be a string path to directory, or False. Saving to home directory instead.")
                plt.savefig( "~/HierarchicalClustering_"+name+imageType, bbox_inches='tight', dpi=dpi)                
        plt.close('all')
        
#%%Calculate difference between two matrices
def calculateDifference(array1, array2, regions, name, saveDir, plot=True, imageType='.pdf', 
                        imageSize=(200,200), fontSize=1, dpi=300, save_matrix=False):
    result = pd.DataFrame(array1-array2)
    result.index=regions
    result.columns=regions
    
    if save_matrix:
        result.to_csv(os.path.join(saveDir, 'Difference_'+name+'.csv'))
        
    if plot:
        plt.Figure(figsize=imageSize)
        sns.heatmap(result, cmap=plt.cm.bwr, vmin=-1.6,vmax=1.6)
        plt.yticks(np.arange(result.shape[0]), result.index.tolist(), fontsize=fontSize)
        plt.xticks(np.arange(result.shape[0]), result.index.tolist(), fontsize=fontSize, rotation='vertical')
    
        
    
        if isinstance(imageType, list):
            for t in imageType:
                try:
                    plt.savefig(os.path.join(saveDir, "Difference_"+name+t), bbox_inches='tight', transparent=True, dpi=dpi)
                except FileNotFoundError:
                    print("Save directory not found. saveDir argument should be a string path to directory, or False. Saving to home directory instead.")
                    plt.savefig( "~/Difference_"+name+t, bbox_inches='tight', transparent=True, dpi=dpi)
                   
        else:
            try:
                plt.savefig(os.path.join(saveDir, "Difference_"+name+imageType), bbox_inches='tight', transparent=True, dpi=dpi)
            except FileNotFoundError:
                print("Save directory not found. saveDir argument should be a string path to directory, or False. Saving to home directory instead.")
                plt.savefig( "~/Difference_"+name+imageType, bbox_inches='tight', dpi=dpi)                
        plt.close('all')