# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 14:40:38 2017

@author: ksburger
"""
import os
import nilearn
from nilearn import plotting
from nilearn import datasets
import numpy as np
from sklearn.svm import SVC
from sklearn.cross_validation import KFold

basepath=os.path.join('/Users','kyle','Desktop','juice2016', 'sub-js02')
# By default 2nd subject will be fetched
haxby_dataset = datasets.fetch_haxby()
fmri_filename = haxby_dataset.func[0] # this is the first instance in the library of a functional scan, fmri_filename is the path to the file
kyle_filename=os.path.join('Users','kyle','Desktop','juice2016', 'sub-js02_task-juicemerge_bold.nii.gz') 
mask_filename = haxby_dataset.mask_vt[0]
kyle_mask=os.path.join(basepath,'brain_mask.nii.gz')
kyle_ana=os.path.join(basepath, 'sub-js02_T1w.nii.gz')
plotting.plot_roi(kyle_mask, bg_img=kyle_ana,cmap='Paired')
