import numpy as np
from enum import Enum


# TODO: add different dictionaries
class DictionaryType(Enum):
    dct_dictionary = 0

class Dictionary:
    def __init__(self, dictionary_type=DictionaryType.dct_dictionary):
        self.dictionary_type = dictionary_type

    def get_dictionary(self, patch_size):
        if patch_size[0] != patch_size[1]:
            raise ValueError('A patch should be square.')

        if self.dictionary_type == DictionaryType.dct_dictionary:
            patch_n = patch_size[0]  # patch size

            dct = np.zeros(patch_size)

            for k in range(patch_n):
                if k > 0:
                    coeff = 1 / np.sqrt(2 * patch_n)
                else:  # k==0
                    coeff = 0.5 / np.sqrt(patch_n)
                dct[:, k] = 2 * coeff * np.cos((0.5 + np.arange(patch_n)) * k * np.pi / patch_n)

            # Create the DCT for both axes
            return np.kron(dct, dct)

#
# def dct_unitary_dictionary(patch_size):
#     """
#     Creating an overcomplete 2D-DCT dictionary.
#     Inputs:
#     patch_size (height, width) tupple (height == width)
#
#     Outputs:
#     dct - Unitary DCT dictionary with normalized columns
#     """
#
#     if patch_size[0] != patch_size[1]:
#         raise ValueError('A patch should be square.')
#
#     patch_n = patch_size[0]  # patch size
#
#     dct = np.zeros(patch_size)
#
#     for k in range(patch_n):
#         if k > 0:
#             coeff=1/np.sqrt(2 * patch_n)
#         else:  # k==0
#             coeff = 0.5 / np.sqrt(patch_n)
#         dct[:, k] = 2 * coeff * np.cos((0.5 + np.arange(patch_n)) * k * np.pi / patch_n)
#
#     # Create the DCT for both axes
#     return np.kron(dct, dct)
#
#


#
# def dct_unitary_dictionary_ref(patch_size):
#     """
#     Creating an overcomplete 2D-DCT dictionary.
#     Inputs:
#     patch_size (height, width) tupple (height == width)
#
#     Outputs:
#     DCT - Unitary DCT dictionary with normalized columns
#     """
#
#     # Make sure that the patch is square
#     try:
#         patch_size[0] != patch_size[1]
#     except:
#         print('This only works for square patches')
#
#     num_atoms = np.prod(patch_size)
#
#     # Create DCT for one axis
#     Pn = int(np.ceil(np.sqrt(num_atoms)))
#     DCT = np.zeros((patch_size[0], Pn))
#
#     for k in range(Pn):
#
#         V = np.cos((0.5 + np.arange(Pn)) * k * np.pi / Pn)
#         if k > 0:
#             V = V - np.mean(V)
#         DCT[:, k] = V / np.sqrt(np.sum(V ** 2))
#
#     # Create the DCT for both axes
#     DCT = np.kron(DCT, DCT)
#
#     return DCT
#

