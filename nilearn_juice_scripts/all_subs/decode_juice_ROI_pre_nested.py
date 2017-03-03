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
#set basepath & add files
basepath=os.path.join('/Users','kylesburger','github','juice2016','all_subs')
fmri_subjs=os.path.join('/Volumes','macX3','juice','w1subjects','prejuice_BOLD_all.nii.gz')
average_ana=os.path.join(basepath,'average_image.nii')
striat_mask=os.path.join(basepath,'fxnl_caudate.nii')

################################################
#load behavoiral data
stim = os.path.join(basepath,'juice_labels_jsall_pre.csv')
labels = np.recfromcsv(stim, delimiter=",")
y = labels['labels']
session = labels['chunk']
print(labels)

#Restrict the analysis to JuiceB and water
#To keep only data corresponding to juiceB or water, we create a
#mask of the samples belonging to the condition.
condition_mask = np.logical_or(y == b'juiceB', y == b'water')
y = y[condition_mask]

#We apply this mask in the sampe direction to restrict the
# classification to the face vs cat discrimination
#We now have less samples
print(y.shape)

#prepare the fxnl data
masker = NiftiMasker(mask_img=striat_mask) #detrend=True, standardize=True) #(add in maker line when for real)
x = masker.fit_transform(fmri_subjs)
x = x[condition_mask]


#plot mask over anatomical that is defined above
#plotting.plot_roi(striat_mask, bg_img=average_ana,cmap='Paired')

#building the decoder
svc = SVC(kernel='linear')
print(svc)

svc.fit(x,y)

coef_ = svc.coef_
print(coef_)
print(coef_.shape)

#k fold example
cv = KFold(n=len(x), n_folds=10)
for train, test in cv:
    svc.fit(x[train], y[train])
    prediction = svc.predict(x[test])
    print((prediction == y[test]).sum() / float(len(y[test])))


from sklearn.grid_search import GridSearchCV
# We are going to tune the parameter 'k' of the step called 'anova' in
# the pipeline. Thus we need to address it as 'anova__k'.

# Note that GridSearchCV takes an n_jobs argument that can make it go
# much faster
grid = GridSearchCV(anova_svc, param_grid={'anova__k': k_range}, verbose=1)
nested_cv_scores = cross_val_score(grid, X, y)


print("Nested CV score: %.4f" % np.mean(nested_cv_scores))

#measuring the chance level with permutation testing.  
from sklearn.cross_validation import permutation_test_score
null_cv_scores = permutation_test_score(svc, fmri_masked, target, cv=cv) 


#taking the coefeicnts from SVC matrix & making it an image
coef_img = masker.inverse_transform(coef_)
print(coef_img)

coef_img.to_filename('jsjuice_svc_weights.nii.gz')

#plotting the coef over the anatomical
from nilearn.plotting import plot_stat_map, show
plot_stat_map(coef_img, bg_img=average_ana, title="SVM weights", display_mode="yx") 
show()