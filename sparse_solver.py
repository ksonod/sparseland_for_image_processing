import numpy as np
from skimage.util import view_as_windows
import numpy.matlib

class SparseSolver:
    def __call__(self, img, dict, sparseland_model):
        patches = self.create_overlapping_patches(img, sparseland_model["patch_size"])
        [est_patches, est_coeffs] = self.batch_thresholding(dict, patches, sparseland_model["epsilon"])
        est_dct = self.col2im(est_patches, sparseland_model["patch_size"], img.shape)

        return est_dct

    def create_overlapping_patches(self, img, patch_size):
        h, w = img.shape

        # number of patches in x and y directions
        x_patch_num = w - patch_size[1] + 1
        y_patch_num = h - patch_size[0] + 1

        # creating overlapping patches (y_patch_num, x_patch_num, patch_size[1], patch_size[0])
        patches = view_as_windows(img, patch_size, step=1)

        # flattened patches
        flattened_patches = patches.reshape(x_patch_num * y_patch_num, np.prod(patch_size)).T

        return flattened_patches

    def batch_thresholding(self, dict, patches, epsilon):
        # BATCH_THRESHOLDING Solve the pursuit problem via the error-constraint
        # Thresholding pursuit
        #
        # Solves the following problem:
        #   min_{alpha_i} \sum_i || alpha_i ||_0
        #                  s.t.  ||y_i - D alpha_i||_2**2 \leq epsilon**2 for all i,
        # where D is a dictionary of size n X n, y_i are the input signals of
        # length n (being the columns of the matrix Y) and epsilon stands
        # for the allowed residual error.
        #
        # The solution is returned in the matrix A, containing the representations
        # of the patches as its columns, along with the denoised signals
        # given by  X = DA.

        # Get the number of atoms
        num_atoms = dict.shape[1]

        # Get the number of patches
        N = patches.shape[1]

        # Compute the inner products between the dictionary atoms and the input patches
        inner_products = np.matmul(dict.T, patches)

        # Compute epsilon**2, which is the square residual error allowed per patch
        epsilon_sq = epsilon ** 2

        # Compute the square value of each entry in 'inner_products' matrix
        residual_sq = inner_products ** 2

        # Sort each column in 'residual_sq' matrix in ascending order
        mat_sorted = np.sort(residual_sq, axis=0)
        mat_inds = np.argsort(residual_sq, axis=0)

        # Compute the cumulative sums for each column of 'mat_sorted' and save the result in the matrix 'accumulate_residual'
        accumulate_residual = np.cumsum(mat_sorted, axis=0)

        # Compute the indices of the dominant coefficients that we want to keep
        inds_to_keep = (accumulate_residual > epsilon_sq)

        # Allocate a matrix of size n X N to save the sparse vectors
        A = np.zeros((num_atoms, N))

        # In what follows we compute the location of each non-zero to be assigned
        # to the matrix of the sparse vectors A. To this end, we need to map
        # 'mat_inds' to a linear subscript format. The mapping will be done using
        # Matlab's 'sub2ind' function.

        # TODO: Replace matlib
        # Create a repetition of the column index for all rows
        col_sub = np.matlib.repmat(np.arange(N), num_atoms, 1)

        # Map the entries in 'inds_to_keep' to their corresponding locations
        # in 'mat_inds' and 'col_sub'.
        mat_inds_to_keep = mat_inds[inds_to_keep]
        col_sub_to_keep = col_sub[inds_to_keep]

        # Assign to 'A' the coefficients in 'inner_products' using
        # the precomputed 'mat_inds_to_keep' and 'col_sub_to_keep'
        A[mat_inds_to_keep, col_sub_to_keep] = inner_products[mat_inds_to_keep, col_sub_to_keep]

        # Reconstruct the patches using 'A' matrix
        X = np.matmul(dict, A)

        return X, A

    # TODO: Edit and clean the code
    def col2im(self, patches, patch_size, im_size):
        # COL_TO_IM Rearrange matrix columns into an image of size MXN
        #
        # Inputs:
        #  patches - A matrix of size p * q, where p is the patch flatten size (height * width = m * n), and q is number of patches.
        #  patch_size - The size of the patch [height width] = [m n]
        #  im_size    - The size of the image we aim to build [height width] = [M N]
        #
        # Output:
        #  im - The reconstructed image, computed by returning the patches in
        #       'patches' to their original locations, followed by a
        #       patch-averaging over the overlaps

        num_im = np.zeros((im_size[0], im_size[1]))
        denom_im = np.zeros((im_size[0], im_size[1]))

        for i in range(im_size[0] - patch_size[0] + 1):
            for j in range(im_size[1] - patch_size[1] + 1):
                # rebuild current patch
                num_of_curr_patch = i * (im_size[1] - patch_size[1] + 1) + (j + 1)
                last_row = i + patch_size[0]
                last_col = j + patch_size[1]
                curr_patch = patches[:, num_of_curr_patch - 1]
                curr_patch = np.reshape(curr_patch, (patch_size[0], patch_size[1]))

                # update 'num_im' and 'denom_im' w.r.t. 'curr_patch'
                num_im[i:last_row, j:last_col] = num_im[i:last_row, j:last_col] + curr_patch
                denom_im[i:last_row, j:last_col] = denom_im[i:last_row, j:last_col] + np.ones(curr_patch.shape)

        # Averaging
        im = num_im / denom_im

        return im