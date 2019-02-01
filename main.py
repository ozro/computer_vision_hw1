import numpy as np
import torchvision
import util
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import visual_words
import visual_recog
import skimage.io

if __name__ == '__main__':

    print("Starting")

    num_cores = util.get_num_CPU()
    # path_img = "../data/kitchen/sun_aasmevtpkslccptd.jpg"
    # image = skimage.io.imread(path_img)
    # image = image.astype('float')/255
    # dictionary = np.load('dictionary.npy')

    ## Display filter responses
    #filter_responses = visual_words.extract_filter_responses(image)
    #util.display_filter_responses(filter_responses)

    ## Compute dictionary
    #visual_words.compute_dictionary(num_workers=num_cores)


    ## Test histograms
    # wordmap = visual_words.get_visual_words(image,dictionary)
    # hist = visual_recog.get_feature_from_wordmap(wordmap,200)
    # plt.bar(np.arange(200),hist)
    # plt.show()

    ## Generate visualized wordmaps
    # files = ["../data/laundromat/sun_aabvooxzwmzzvwds.jpg",
    #          "../data/laundromat/sun_aaprcnhpdrhlnhji.jpg",
    #          "../data/laundromat/sun_aalvewxltowiudlw.jpg"]
    # i = 0
    # for path in files:
    #     image = skimage.io.imread(path)
    #     image = image.astype('float')/255
    #     wordmap = visual_words.get_visual_words(image,dictionary)
    #     util.save_wordmap(wordmap, "wordmap{}".format(i))
    #     i += 1

    ## Build recognition system
    visual_recog.build_recognition_system(num_workers=num_cores)

    #conf, accuracy = visual_recog.evaluate_recognition_system(num_workers=num_cores)
    #print(conf)
    #print(np.diag(conf).sum()/conf.sum())

