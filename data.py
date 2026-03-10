from dipy.io.image import load_nifti, load_nifti_data
from dipy.data import default_sphere, get_fnames
from dipy.io.gradients import read_bvals_bvecs

hardi_fname, hardi_bval_fname, hardi_bvec_fame= get_fnames(name="stanford_hardi")
label_fname =get_fnames(name="stanford_labels")

data, affine, hardi_img, vox_size= load_nifti(hardi_fname, return_img=True, return_voxsize=True)
labels = load_nifti_data(label_fname)
bvals, bvecs = read_bvals_bvecs(hardi_bval_fname,hardi_bvec_fame)
