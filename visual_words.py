from multiprocessing import Pool 
import os
import random
import time

import imageio
import numpy as np
import scipy.ndimage
import scipy.spatial.distance
import skimage.color
import sklearn.cluster

import util

def extract_filter_responses(image):
    '''
    Extracts the filter responses for the given image.

    [input]
    * image: numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''
    
    if len(image.shape) == 2:
        image = np.tile(image[:, np.newaxis], (1, 1, 3))

    if image.shape[2] == 4:
        image = image[:,:,0:3]

    image = skimage.color.rgb2lab(image)
    
    scales = [1,2,4,8,8*np.sqrt(2)]
    for i in range(len(scales)):
        for c in range(3):
            #img = skimage.transform.resize(image, (int(ss[0]/scales[i]),int(ss[1]/scales[i])),anti_aliasing=True)
            img = scipy.ndimage.gaussian_filter(image[:,:,c],sigma=scales[i])
            if i == 0 and c == 0:
                imgs = img[:,:,np.newaxis]
            else:
                imgs = np.concatenate((imgs,img[:,:,np.newaxis]),axis=2)
        for c in range(3):
            img = scipy.ndimage.gaussian_laplace(image[:,:,c],sigma=scales[i])
            imgs = np.concatenate((imgs,img[:,:,np.newaxis]),axis=2)
        for c in range(3):
            img = scipy.ndimage.gaussian_filter(image[:,:,c],sigma=scales[i],order=[0,1])
            imgs = np.concatenate((imgs,img[:,:,np.newaxis]),axis=2)
        for c in range(3):
            img = scipy.ndimage.gaussian_filter(image[:,:,c],sigma=scales[i],order=[1,0])
            imgs = np.concatenate((imgs,img[:,:,np.newaxis]),axis=2)

    return imgs

def get_visual_words(image,dictionary):
    '''
    Compute visual words mapping for the given image using the dictionary of visual words.

    [input]
    * image: numpy.ndarray of shape (H,W) or (H,W,3)
    
    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''
    
    # ----- Implementation -----
    filter_responses = extract_filter_responses(image)
    shape = filter_responses.shape
    responses = np.reshape(filter_responses, (np.prod(shape[0:2]), filter_responses.shape[2]))
    distances = scipy.spatial.distance.cdist(responses, dictionary, "euclidean")
    mins = np.argmin(distances, axis = 1)
    mins = np.reshape(mins, shape[0:2])#/dictionary.shape[0]
    return mins


def compute_dictionary_one_image(args):
    '''
    Extracts random samples of the dictionary entries from an image.
    This is a function run by a subprocess.

    [input]
    * i: index of training image
    * alpha: number of random samples
    * image_path: path of image file
    * tmpdirname: path of temp directory

    [saved]
    * sampled_response: numpy.ndarray of shape (alpha,3F)
    '''

    i,alpha,image_path,tmpdirname = args

    # ----- Implementation -----
    image_path = os.path.join("..", "data", image_path)
    print("#{}: Processing image at {}".format(i, image_path))
    image = skimage.io.imread(image_path)
    image = image.astype('float')/255
    filter_responses = extract_filter_responses(image)

    # Generate random mask and sample alpha responses
    x = [True] * alpha + [False] * (np.prod(filter_responses.shape[0:2]) - alpha)
    np.random.shuffle(x)
    mask = np.reshape(np.asarray(x), filter_responses.shape[0:2])
    response = filter_responses[mask,:]
    
    tmpfilename = os.path.join(tmpdirname, "{}".format(i)) 
    print("#{}: Saving response with shape {} at {}".format(i, response.shape, tmpfilename))
    np.save(tmpfilename, response)

def compute_dictionary(num_workers=2):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * num_workers: number of workers to process in parallel
    
    [saved]
    * dictionary: numpy.ndarray of shape (K,3F)
    '''

    train_data = np.load("../data/train_data.npz")

    # ----- Implementation -----

    # Create temp dir for results
    tmpdirname = os.path.join("..", "responses")

    # Generate args
    K = 100
    alpha = 500
    filter_count = 20

    files = train_data['files']
    num_files = len(files)
    print("\nStarted dictionary computation with temporary directory:", tmpdirname)
    print("Starting pool of {} workers for {} files\n".format(num_workers, files.size))
    
    # Start subprocess
    pool = Pool(processes=num_workers)
    i = 0
    for file_path in files:
        pool.apply_async(compute_dictionary_one_image, ((i,alpha,file_path,tmpdirname),))
        i+=1
    pool.close()
    pool.join()


    print("\n Gathering Results")
    results = np.empty((alpha * num_files, 3*filter_count))
    i = 0
    for filename in os.listdir(tmpdirname):
        print("Adding {} to index {}:{}".format(os.path.join(tmpdirname, filename), i*alpha, (i+1) * alpha))
        results[i*alpha:(i+1)*alpha, :] = np.load(os.path.join(tmpdirname, filename))
        i += 1
    
    print("\n Computing k-means with n_clusters = {}, n_jobs = {}, data shape = {}".format(K, num_workers, results.shape))
    kmeans = sklearn.cluster.KMeans(n_clusters=K, n_jobs=num_workers).fit(results)
    dictionary = kmeans.cluster_centers_
    savefile = "dictionary.npy"
    print("Saving results to: {}".format(savefile))
    np.save(savefile, dictionary)