#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 00:31:58 2019

@author: smith
"""

import pandas as pd
import os
import numpy as np
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from scipy.stats import zscore

baseDirectory = '/d2/studies/SmartSPIM/Kroener/analysis/collapsed'
os.chdir(baseDirectory)
date = 'May5'

regionIndex = pd.read_csv('/d2/studies/ClearMap/RegionID_Index.csv', index_col=0, encoding='ISO-8859-1')
regionVolumes = pd.read_csv('/d2/studies/ClearMap/RegionVolumeIndex.csv', index_col=0, encoding='ISO-8859-1')


r_veh = ['FtdTom_1NP', 'FtdTom_1RB', 'FtdTom_1LB', 'FtdTom_3NP', 'FtdTom_4RB', 'FtdTom_5RT', 'FtdTom_5LT']
ext = ['FtdTom_1LT',  'FtdTom_2NP', 'FtdTom_2LB', 'FtdTom_3LT', 'FtdTom_4NP', 'FtdTom_4RT', 'FtdTom_4LB']
r_tam = ['FtdTom_1RT', 'FtdTom_2RT', 'FtdTom_2RB', 'FtdTom_2LT', 'FtdTom_3RB', 'FtdTom_3LB', 'FtdTom_4LT', 'FtdTom_5NP']

cat_veh = pd.DataFrame()

for sample in r_veh:
    df = pd.read_csv(os.path.join(baseDirectory, sample+'_Ilastik/'+sample+ '_Ilastik_Annotated_Counts_Processed.csv'), error_bad_lines=False, index_col=0)
    df.drop(['name', 'Volume', 'Count_Density', 'Zscore', 'Oprm1_Expression'], axis=1, inplace=True)
    #  df.set_index('id',inplace=True)
    cat_veh = pd.concat([cat_veh, df],axis=1)
cat_veh.columns=r_veh

cat_veh['region']=regionIndex['name']
cat_veh['acronym']=regionIndex['acronym']
cat_veh['Volume']=regionVolumes['Volume']
for sample in r_veh:
    cat_veh[sample+'_VolNorm']=cat_veh[sample]/cat_veh['Volume']
cat_veh.fillna(value=0, inplace=True)
cat_veh['MeanDensity']=np.mean(cat_veh.iloc[:,-(len(r_veh)):], axis=1)
cat_veh.sort_values(by='MeanDensity', ascending=False, inplace=True)
cat_veh.to_csv(os.path.join(baseDirectory, 'Vehicle_Counts_'+date+'.csv'))


cat_tam = pd.DataFrame()

for sample in r_tam:
    df = pd.read_csv(os.path.join(baseDirectory, sample+'_Ilastik/'+sample+ '_Ilastik_Annotated_Counts_Processed.csv'), error_bad_lines=False, index_col=0)
    df.drop(['name', 'Volume', 'Count_Density', 'Zscore'], axis=1, inplace=True)
    #  df.set_index('id',inplace=True)
    cat_tam = pd.concat([cat_tam, df],axis=1)
cat_tam.columns=r_tam
    

cat_tam['region']=regionIndex['name']
cat_tam['acronym']=regionIndex['acronym']
cat_tam['Volume']=regionVolumes['Volume']
for sample in r_tam:
    cat_tam[sample+'_VolNorm']=cat_tam[sample]/cat_tam['Volume']
cat_tam.fillna(value=0, inplace=True)
cat_tam['MeanDensity']=np.mean(cat_tam.iloc[:,-(len(r_tam)):], axis=1)
cat_tam.sort_values(by='MeanDensity', ascending=False, inplace=True)
cat_tam.to_csv(os.path.join(baseDirectory, 'Reinstated_4OHT_Counts_'+date+'.csv'))


cat_ext = pd.DataFrame()

for sample in ext:
    df = pd.read_csv(os.path.join(baseDirectory, sample+'_Ilastik/'+sample+ '_Ilastik_Annotated_Counts_Processed.csv'), error_bad_lines=False, index_col=0)
    df.drop(['name', 'Volume', 'Count_Density', 'Zscore'], axis=1, inplace=True)
    #  df.set_index('id',inplace=True)
    cat_ext = pd.concat([cat_ext, df],axis=1)
cat_ext.columns=ext

cat_ext['region']=regionIndex['name']
cat_ext['acronym']=regionIndex['acronym']
cat_ext['Volume']=regionVolumes['Volume']
for sample in ext:
    cat_ext[sample+'_VolNorm']=cat_ext[sample]/cat_ext['Volume']
cat_ext.fillna(value=0, inplace=True)
cat_ext['MeanDensity']=np.mean(cat_ext.iloc[:,-(len(ext)):], axis=1)
cat_ext.sort_values(by='MeanDensity', ascending=False, inplace=True)
cat_ext.to_csv(os.path.join(baseDirectory, 'Extinguished_4OHT_Counts_'+date+'.csv'))


cat_all = pd.DataFrame()
samples = ['FtdTom_1NP', 'FtdTom_1RB', 'FtdTom_1LB', 'FtdTom_3NP', 'FtdTom_4RB', 'FtdTom_5RT', 'FtdTom_5LT',
           'FtdTom_1LT',  'FtdTom_2NP', 'FtdTom_2LB', 'FtdTom_3LT', 'FtdTom_4NP', 'FtdTom_4RT', 'FtdTom_4LB', 
           'FtdTom_1RT', 'FtdTom_2RT', 'FtdTom_2RB', 'FtdTom_2LT', 'FtdTom_3RB', 'FtdTom_3LB', 'FtdTom_4LT', 'FtdTom_5NP']

for mouse in samples:
    if mouse in ext:
        cat_all[mouse]=cat_ext[mouse]
    elif mouse in r_veh:
        cat_all[mouse]=cat_veh[mouse]
    elif mouse in r_tam:
        cat_all[mouse]=cat_tam[mouse]
cat_all.fillna(0, inplace=True)
vehData = cat_all.iloc[:,:len(r_veh)]
extData = cat_all.iloc[:,len(r_veh):len(r_veh)+len(ext)]
reinstData = cat_all.iloc[:,-(len(r_tam)):]
pvalsRE = np.array([ttest_ind(cat_all.iloc[i,len(r_veh):len(r_veh)+len(ext)], cat_all.iloc[i,-(len(r_tam)):], nan_policy='omit')[1] for i in range(reinstData.shape[0])])
cat_all['pval_RvsE']=pvalsRE
pvalsEV = np.array([ttest_ind(cat_all.iloc[i,:len(r_veh)], cat_all.iloc[i,len(r_veh):len(r_veh)+len(ext)], nan_policy='omit')[1] for i in range(reinstData.shape[0])])
cat_all['pval_EV']=pvalsEV
pvalsRV = np.array([ttest_ind(cat_all.iloc[i,:len(r_veh)], cat_all.iloc[i,-(len(r_tam)):], nan_policy='omit')[1] for i in range(reinstData.shape[0])])
cat_all['pval_RV']=pvalsRV

pvals_EMF =  np.array([ttest_ind(extData.iloc[i,:3], extData.iloc[i,3:], nan_policy='omit')[1] for i in range(extData.shape[0])])
pvals_RMF =  np.array([ttest_ind(reinstData.iloc[i,:4], extData.iloc[i,4:], nan_policy='omit')[1] for i in range(reinstData.shape[0])])

cat_all['pval_Ext_MaleFemale']=pvals_EMF
cat_all['pval_Reinstated_MaleFemale']=pvals_RMF

cat_all.sort_values(by='pval_RvsE', ascending=True, inplace=True)

cat_all['region']=regionIndex['name']
cat_all['acronym']=regionIndex['acronym']
cat_all['volume']=regionVolumes['Volume']

cat_all.to_csv(os.path.join(baseDirectory, 'CombinedData_Stats_'+date+'_FullDetailed.csv'))

cat_vol = pd.DataFrame()
mice = cat_all.columns[:-8]
for mouse in mice:
    cat_vol[mouse]=cat_all[mouse]/cat_all['volume']
pvalsRE_vol = np.array([ttest_ind(cat_vol.iloc[i,len(r_veh):len(r_veh)+len(ext)], cat_vol.iloc[i,-(len(r_tam)):], nan_policy='omit')[1] for i in range(cat_vol.shape[0])])
pvalsEV_vol = np.array([ttest_ind(cat_vol.iloc[i,:len(r_veh)], cat_vol.iloc[i,len(r_veh):len(r_veh)+len(ext)], nan_policy='omit')[1] for i in range(cat_vol.shape[0])])
pvalsRV_vol = np.array([ttest_ind(cat_vol.iloc[i,:len(r_veh)], cat_vol.iloc[i,-(len(r_tam)):], nan_policy='omit')[1] for i in range(cat_vol.shape[0])])

cat_vol['region']=regionIndex['name']
vehData_vol = cat_vol.iloc[:,:len(r_veh)]
extData_vol = cat_vol.iloc[:,len(r_veh):len(r_veh)+len(ext)]
reinstData_vol = cat_vol.iloc[:,len(r_veh)+len(ext):]

cat_vol['Vehicle_Mean'] = np.mean(vehData_vol, axis=1)
cat_vol['Extinguished_Mean'] = np.mean(extData_vol, axis=1)
cat_vol['Reinstated_Mean'] = np.mean(reinstData_vol, axis=1)

cat_vol['pval_RvsE']=pvalsRE_vol
cat_vol['pval_EvsV']=pvalsEV_vol
cat_vol['pval_RvsV']=pvalsRV_vol
cat_vol['acronym']=regionIndex['acronym']
cat_vol.sort_values(by='pval_RvsE', ascending=True, inplace=True)
cat_vol.to_csv(os.path.join(baseDirectory, 'CombinedData_VolumeNormalized_'+date+'_FullDetailed.csv'))

data = pd.read_csv('/d2/studies/ClearMap/TRAP2_tdTomato_iDISCO/Cohort1_IlastikAnalysis/CombinedData_VolumeNormalized_April19_FullDetailed.csv', index_col=0)
data['Reinst_Density_Zscore']=zscore(np.array(data['Reinstated_Mean']), axis=0)
data['Oprm1_Zscore']=zscore(np.array(data['Oprm1_Expression']), axis=0, nan_policy='omit')
data['Zscore_RvsE']=zscore(np.array(data['pval_RvsE']), axis=0, nan_policy='omit')

data.to_csv('/d2/studies/ClearMap/TRAP2_tdTomato_iDISCO/Cohort1_IlastikAnalysis/CombinedData_VolumeNormalized_April19__FullDetailed_wOprm1.csv')




