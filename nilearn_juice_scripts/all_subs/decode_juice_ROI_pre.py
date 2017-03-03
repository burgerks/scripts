import os
import nilearn
from nilearn import plotting
from nilearn.input_data import NiftiMasker
import numpy as np
from sklearn.svm import SVC
from sklearn.cross_validation import KFold

#below is an example of k-fold decoding of two subjects from juice study comparing 
#juiceB receipt vs. rest using left caudate ROI


####################
#set basepath
basepath=os.path.join('/Users','kyle','Desktop','juice2016','all_subs')


#######################################
#load & prepare MRI data
   
#prepare the fxnl data
masker = NiftiMasker(mask_img=striat_mask, detrend=True,standardize=True)
fmri_trans = masker.fit_transform(fmri_subjs)
print(fmri_trans)

#load, fxnl, anatomical & mask for plotting & funsies
fmri_subjs=os.path.join('/Volumes','macX3','juice','w1subjects','prejuice_BOLD_all.nii.gz') 
average_ana=os.path.join(basepath,'average_image.nii')
striat_mask=os.path.join('/Users','kyle','Desktop','juice2016','masks','mask_roi197.nii')

#saving the detrended and normalized
outfile=os.path.join(basepath,'prejuice_BOLD_de-norm.nii.gz')
np.save(outfile,fmri_trans)
infile=os.path.join(basepath,'prejuice_BOLD_de-norm.nii.gz.npy')
X=np.load(infile)
print(X)

#plot mask over anatomical that is defined above
plotting.plot_roi(striat_mask, bg_img=average_ana,cmap='Paired')


################################################
#load behavoiral data
stim = os.path.join(basepath,'juice_labels_jsall_pre.csv')
labels = np.recfromcsv(stim, delimiter=",")
print(labels)

#Retrieve the behavioral targets, that we are going to predict in the decoding
target = labels['labels']
print(target)

#Restrict the analysis to JuiceB and rest
#To keep only data corresponding to juiceB or water, we create a
#mask of the samples belonging to the condition.
condition_mask = np.logical_or(target == b'juiceA', target == b'water')
fmri_masked = fmri_trans[condition_mask]

#We apply this mask in the sampe direction to restrict the
# classification to the face vs cat discrimination
#We now have less samples
print(fmri_masked.shape)

#We apply the same mask to the targets
target = target[condition_mask]
print(target.shape)

svc = SVC(kernel='linear')
print(svc)
svc.fit(fmri_masked, target)

#k fold example
cv = KFold(n=len(fmri_masked), n_folds=10)
for train, test in cv:
    svc.fit(fmri_masked[train], target[train])
    prediction = svc.predict(fmri_masked[test])
    print((prediction == target[test]).sum() / float(len(target[test])))

#permunation testing to measure probablility of chance
from sklearn.cross_validation import permutation_test_score
null_cv_scores = permutation_test_score(svc, fmri_masked, target, cv=cv)  
print(null_cv_scores)



coef_ = svc.coef_
print(coef_)
print(coef_.shape)

#taking the coefeicnts from SVC matrix & making it an image
coef_img = masker.inverse_transform(coef_)
print(coef_img)

coef_img.to_filename('jsjuice_svc_weights.nii.gz')

#plotting the coef over the anatomical
from nilearn.plotting import plot_stat_map, show
plot_stat_map(coef_img, bg_img=average_ana, title="SVM weights", display_mode="yx") 
show()

