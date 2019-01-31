import numpy as np
import threading
import queue
import imageio
import os,time
import math
import visual_words

def build_recognition_system(num_workers=2):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * num_workers: number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''



    train_data = np.load("../data/train_data.npz")
    dictionary = np.load("dictionary.npy")
    # ----- TODO -----

    pass

def evaluate_recognition_system(num_workers=2):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * num_workers: number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''


    test_data = np.load("../data/test_data.npz")
    trained_system = np.load("trained_system.npz")
    # ----- TODO -----
    pass




def get_image_feature(file_path,dictionary,layer_num,K):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * file_path: path of image file to read
    * dictionary: numpy.ndarray of shape (K,3F)
    * layer_num: number of spatial pyramid layers
    * K: number of clusters for the word maps

    [output]
    * feature: numpy.ndarray of shape (K)
    '''
    pass


    # ----- TODO -----


def distance_to_set(word_hist,histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * sim: numpy.ndarray of shape (N)
    '''
    pass
    


    # ----- TODO -----



def get_feature_from_wordmap(wordmap,dict_size):
    '''
    Compute histogram of visual words.

    [input]
    * wordmap: numpy.ndarray of shape (H,W)
    * dict_size: dictionary size K

    [output]
    * hist: numpy.ndarray of shape (K)
    '''
    
    # ----- Implemented -----
    hist,_ = np.histogram(wordmap, bins=np.arange(dict_size+1), density=True)
    return hist

def get_feature_from_wordmap_SPM(wordmap,layer_num,dict_size):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * wordmap: numpy.ndarray of shape (H,W)
    * layer_num: number of spatial pyramid layers
    * dict_size: dictionary size K

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^layer_num-1)/3)
    '''
    
    # ----- Implementation -----
    hist_all = np.zeros(int(dict_size*(4**(layer_num)-1)/3))
    base_weight = 0.5

    i = 0
    for curr_layer in range(layer_num-1, -1, -1):
        if(curr_layer == layer_num-1): #Smallest cell size, calculate histograms
            for cell in ndarray_split(wordmap, 2**curr_layer):
                hist = get_feature_from_wordmap(cell, dict_size)
                hist_all[i:i+dict_size] = hist * base_weight
                i+=dict_size
        else: #Construct histograms from previous layer
            prev_len = 2**(curr_layer+1)
            curr_len = 2**curr_layer

            prev_inds = prev_len*prev_len*dict_size
            prev_hists = hist_all[i-prev_inds:i]
            for j in range(curr_len*curr_len):
                row,col = np.unravel_index(j, (curr_len, curr_len))
                start1 = np.ravel_multi_index((row*2,col*2*dict_size), (prev_len, prev_len*dict_size)) 
                start2 = start1 + dict_size * prev_len
                if(curr_layer != 0):
                    weight = 0.5
                else:
                    weight = 1
                #print("Construction histogram at ({},{}) at indices {},{},{} and {}".format(row, col, start1, start1+dict_size, start2, start2 + dict_size))
                hist1 = (prev_hists[start1:start1+dict_size])
                hist2 = (prev_hists[start1 + dict_size:start1+dict_size*2])
                hist3 = (prev_hists[start2:start2+dict_size])
                hist4 = (prev_hists[start2 + dict_size:start2+dict_size*2])
                # print(hist1)
                # print(hist2)
                # print(hist3)
                # print(hist4)
                # print(hist1+hist2+hist3+hist4)
                hist_all[i:i+dict_size] = (hist1+hist2+hist3+hist4) / 4 * weight

                i+=dict_size
    return hist_all

def ndarray_split(a, n):
    '''
    Split a 2D array into n*n cells, evenly sized.

    [input]
    * a: numpy.ndarray of 2 dimensions
    * n: number of splits per side

    [output]
    * split_array: array of 2D cells
    '''

    split_array = []
    vsplit = np.array_split(a, n, axis=0)
    i = 0
    for row in vsplit:
        split_array.extend(np.array_split(row, n, axis=1))
        i += 1

    return np.asarray(split_array)
