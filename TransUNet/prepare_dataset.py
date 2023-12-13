import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
import os
import sys
import time
import math
from glob import glob
import SimpleITK as sitk
from scipy import ndimage
from scipy import misc
import cv2
from tqdm import tqdm
from natsort import natsorted
import h5py
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors




def read_mhd_and_raw(mhd_file_path):
    # returns the image array, origin and spacing of a .mhd image file
    # image aray is a numpy array containing the respective image
    # origin is a numpy array containing the x,y,z coordinates of the origin of the image
    # spacing is a numpy array containing the pixel spacing


    itkimage = sitk.ReadImage(mhd_file_path)
    image_array = sitk.GetArrayFromImage(itkimage)
    origin = np.array(itkimage.GetOrigin())
    spacing = np.array(itkimage.GetSpacing())
    return image_array, origin, spacing

def get_mhd_image_at_index(mhd_files, index):
    mhd_file_path = mhd_files[index]
    image_array, origin, spacing = read_mhd_and_raw(mhd_file_path)
    return image_array, origin, spacing


def convert_mhd_to_npz(image_mhd_files, label_mhd_files, npz_folder_path):
    assert len(image_mhd_files) == len(label_mhd_files)
    for i in tqdm(range(len(image_mhd_files))):
        # read image
        im = get_mhd_image_at_index(image_mhd_files, i)[0]
        #upsample image to 512x512
        im =  cv2.resize(im, (512, 512), interpolation = cv2.INTER_NEAREST)
        # normalize image
        im = (im - np.min(im)) / (np.max(im) - np.min(im))
        # convert to float32
        im = im.astype(np.float32)

        # read label
        im_label = get_mhd_image_at_index(label_mhd_files, i)[0]
        #upsample image to 512x512
        im_label =  cv2.resize(im_label, (512, 512), interpolation = cv2.INTER_NEAREST)
        # convert to float32
        im_label = im_label.astype(np.float32)

        # convert them to npz
        npz_name = os.path.basename(image_mhd_files[i]).split(".")[0]
        save_path = os.path.join(npz_folder_path, f'{npz_name}.npz')
        np.savez(save_path, image=im, label=im_label)
        


def classify_test_to_cases(test_npz_files, save_folder=None):
    # classify test npz files to cases
    cases = [[]]
    path_prefix = os.path.dirname(test_npz_files[0])
    last_number = int(os.path.basename(test_npz_files[0]).split(".")[0].split("_")[1])
    for npz_file in test_npz_files:
        cur_name = os.path.basename(npz_file).split(".")[0]
        cur_number = int(cur_name.split("_")[1])
        if cur_number - last_number <= 1:
            cases[-1].append(os.path.basename(npz_file))
        else:
            cases.append([os.path.basename(npz_file)])
        last_number = cur_number

    # # rename cases by adding casexxx_ prefix
    # for i in range(len(cases)):
    #     for j in range(len(cases[i])):
    #         cases[i][j] = f'case{i+1:03d}_' + cases[i][j]


    # load npz files for each case
    cases_npz = []
    for case in cases:
        case_npz = []
        for npz_file in case:
            case_npz.append(np.load(os.path.join(path_prefix, npz_file)))
        cases_npz.append(case_npz)


    # convert each case to npz.h5 file
    list_names = []
    for i in range(len(cases_npz)):
        case_npz_image = []
        for npz in cases_npz[i]:
            case_npz_image.append(npz['image'])
        case_npz_image = np.array(case_npz_image)
        
        case_npz_label = []
        for npz in cases_npz[i]:
            case_npz_label.append(npz['label'])
        case_npz_label = np.array(case_npz_label)

        
        # save to h5 file
        if save_folder is not None:
            save_path = os.path.join(save_folder, f'case{i+1:04d}.npy.h5')
            list_names.append(f'case{i+1:04d}')
            with h5py.File(save_path, 'w') as f:
                f.create_dataset('image', data=case_npz_image)
                f.create_dataset('label', data=case_npz_label)
                f.close()

    list_file_path = "../data/tumor/lists/test_cases.txt"
    with open(list_file_path, 'w') as f:
        for item in list_names:
            f.write("%s\n" % item)
    
    return cases



if __name__ == "__main__":

    print("Converting testing data...")
    test_image_mhd_files = glob('../data/tumor/Testing/Brains/*.mhd')
    test_label_mhd_files = glob('../data/tumor/Testing/Labels/*.mhd')
    test_image_mhd_files = natsorted(test_image_mhd_files)
    test_label_mhd_files = natsorted(test_label_mhd_files)

    convert_mhd_to_npz(test_image_mhd_files, test_label_mhd_files, '../data/tumor/Testing')
    # save a list.txt file that contains name of all test npz files
    test_npz_names = [os.path.basename(npz_file).split(".")[0] for npz_file in glob('../data/tumor/Testing/*.npz')]

    print("indivisual npz files are saved to ../data/tumor/Testing")


    # save to list.txt
    test_npz_names = natsorted(test_npz_names)
    with open('../data/tumor/lists/test.txt', 'w') as f:
        for item in test_npz_names:
            f.write("%s\n" % item)

    print("list.txt is saved to ../data/tumor/lists/test.txt")


    test_npz_files = glob('../data/tumor/Testing/*.npz')
    test_npz_files = natsorted(test_npz_files)
    cases = classify_test_to_cases(test_npz_files, save_folder='../data/tumor/Testing_cases')

    print("cases are saved to ../data/tumor/Testing_cases")
    print("list.txt is saved to ../data/tumor/lists/test_cases.txt")

    print("Done!")

