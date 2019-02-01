import numpy as np
import threading
import queue
import imageio
import os,time
import math
import visual_words
import skimage.io
import matplotlib.pyplot as plt
from multiprocessing.pool import ThreadPool as Pool


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

    # ----- Implementation -----
    file_paths = train_data['files']
    labels = train_data['labels']
    SPM_layer_num = 3
    K = dictionary.shape[0]
    save_dir = os.path.join("..","features")
    feature_size = int(K*(4**SPM_layer_num-1)/3)
    features = np.zeros((labels.size, feature_size))

    i = 0
    pool = Pool(num_workers)
    for file_path in file_paths:
        pool.apply_async(get_image_feature_worker, (file_path,dictionary,SPM_layer_num,K,i,save_dir))
        i+=1
    pool.close()
    pool.join()

    i = 0
    for filename in os.listdir(save_dir):
        print("Adding {} to index {}".format(os.path.join(save_dir, filename), i))
        features[i,:] = np.load(os.path.join(save_dir, filename))
        i += 1
    np.savez("trained_system", dictionary=dictionary, features=features, labels=labels, SPM_layer_num=SPM_layer_num)

def get_image_feature_worker(file_path, dictionary, SPM_layer_num, K, index, save_dir):
    '''
    Worker for asynchronous extraction of image feature vector`

    [input]
    * file_path: path to the image 
    * dictionary: trained dictionary
    * SPM_layer_num: number of layers in spatial pyramid
    * K: number of entries in dictionary
    * index: index of current image
    * save_dir: directory to save results to 

    [saved]
    * feature: SPM histogram vector for image 
    '''

    print("#{}: Processing image at {}".format(index, file_path))
    feature = get_image_feature(os.path.join("..", "data", file_path), dictionary, SPM_layer_num, K)
    tmpfilename = os.path.join(save_dir, "{}".format(index)) 
    print("#{}: Saving response with shape {} at {}".format(index, feature.shape, tmpfilename))
    np.save(tmpfilename, feature)

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
    # ----- Implementation -----
    test_files = test_data['files']
    test_labels = test_data['labels']

    SPM_layer_num = trained_system['SPM_layer_num']
    trained_features = trained_system['features']
    trained_labels = trained_system['labels']
    dictionary = trained_system['dictionary']
    K = dictionary.shape[0]

    result_labels = np.zeros(test_labels.size)
    i = 0
    pool = Pool(num_workers)
    for file_path in test_files:
        pool.apply_async(evaluation_worker, args = (file_path,dictionary,SPM_layer_num,K,i, trained_features, trained_labels, result_labels), callback = evaluation_callback)
        i+=1
    pool.close()
    pool.join()

    print("Constructing confusion matrix")

    conf = np.zeros((8,8))
    for i in range(len(result_labels)):
        eval_label = int(result_labels[i])
        true_label = test_labels[i]
        print("Test {} classified {} as {}".format(i, true_label, eval_label))
        conf[true_label][eval_label] += 1
    accuracy = np.diag(conf).sum()/conf.sum()

    return(conf, accuracy)


def evaluation_worker(file_path, dictionary, SPM_layer_num, K, index, trained_features, trained_labels, result_labels):
    print("#{}: Processing image at {}".format(index, file_path))
    feature = get_image_feature(os.path.join("..", "data", file_path), dictionary, SPM_layer_num, K)
    print("#{}: Finding best label from feature of shape {}".format(index, feature.shape))
    sim = distance_to_set(feature, trained_features)
    max_index = sim.argmax(axis=0)
    label = trained_labels[max_index]

    return (index, label, result_labels)


def evaluation_callback(result):
    index, label, result_labels = result
    print("Saving label {} at index {}".format(label, index))
    result_labels[index] = label

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


    # ----- Implemented -----
    image = skimage.io.imread(file_path)
    image = image.astype('float')/255

    wordmap = visual_words.get_visual_words(image,dictionary)
    return get_feature_from_wordmap_SPM(wordmap, layer_num, K)


def distance_to_set(word_hist,histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * sim: numpy.ndarray of shape (N)
    '''

    # ----- Implementation -----
    hists = np.tile(word_hist, (histograms.shape[0], 1))
    return np.sum(np.minimum(hists, histograms), axis=1)

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

    #Construct smallest layer histograms
    i = 0
    vsplit = np.array_split(wordmap, 2**(layer_num -1), axis=0)
    for row in vsplit:
        cells = np.array_split(row, 2**(layer_num -1), axis=1)
        for cell in cells:
            hist = get_feature_from_wordmap(cell, dict_size)
            hist_all[i:i+dict_size] = hist * base_weight
            i+=dict_size
    
    #Construct histograms from previous layer
    for curr_layer in range(layer_num-2, -1, -1):
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
            hist1 = (prev_hists[start1:start1+dict_size])
            hist2 = (prev_hists[start1 + dict_size:start1+dict_size*2])
            hist3 = (prev_hists[start2:start2+dict_size])
            hist4 = (prev_hists[start2 + dict_size:start2+dict_size*2])
            hist_all[i:i+dict_size] = (hist1+hist2+hist3+hist4) / 4 * weight

            i+=dict_size
    return hist_all