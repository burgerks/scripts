import os
import nilearn
from nilearn import datasets
from nilearn import plotting
from nilearn.input_data import NiftiMasker
import numpy as np
from sklearn.svm import SVC
from sklearn.cross_validation import KFold

####################
#set basepath
basepath=os.path.join('/Users','kyle','Desktop','juice2016','sub-js02')

#######################################
#Prepare and load MRI data
fMRI_subjs02=os.path.join(basepath,'sub-js02_task-juicemerge_bold.nii.gz') 

#load mask & anatomical for plotting & funsies
kyle_mask=os.path.join('/Users','kyle','Desktop','juice2016','masks','mask_roi197.nii')
kyle_ana=os.path.join(basepath, 'sub-js02_T1w.nii.gz')
 
#plot mask over anatomical that is defined above
plotting.plot_roi(kyle_mask, bg_img=kyle_ana,cmap='Paired')

masker = NiftiMasker(mask_img=kyle_mask, standardize=True)
fMRI_masked = masker.fit_transform(fMRI_subjs02)
print(fMRI_masked)

################################################
#load behavoiral data
stim = os.path.join(basepath, 'labels_js02.csv')
labels = np.recfromcsv(stim, delimiter=",")
print(labels)

# Retrieve the behavioral targets, that we are going to predict in the
# decoding
target = labels['labels']
print(target)

# Restrict the analysis to JuiceA and water
# To keep only data corresponding to juiceA or water, we create a
# mask of the samples belonging to the condition.
condition_mask = np.logical_or(target == b'juiceB', target == b'rest')

# We apply this mask in the sampe direction to restrict the
# classification to the face vs cat discrimination
fMRI_masked = fMRI_masked[condition_mask]

# We now have less samples
print(fMRI_masked.shape)

# We apply the same mask to the targets
target = target[condition_mask]
print(target.shape)


from sklearn.svm import SVC
svc = SVC(kernel='linear')
print(svc)

from sklearn.cross_validation import KFold
cv = KFold(n=len(fMRI_masked), n_folds=10)

for train, test in cv:
    svc.fit(fMRI_masked[train], target[train])
    prediction = svc.predict(fMRI_masked[test])
    print((prediction == target[test]).sum() / float(len(target[test])))


    
from sklearn.cross_validation import cross_val_score
cv_score = cross_val_score(svc, fMRI_masked, target)
print(cv_score)

coef_ = svc.coef_
print(coef_)
print(coef_.shape)

coef_img = masker.inverse_transform(coef_)
print(coef_img)


coef_img.to_filename('js02juice_svc_weights.nii.gz')

from nilearn.plotting import plot_stat_map, show

plot_stat_map(coef_img, bg_img=kyle_ana,
              title="SVM weights", display_mode="yx")
              show()