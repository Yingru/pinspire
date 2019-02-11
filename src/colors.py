import numpy as np
import cv2
from sklearn.cluster import KMeans
import os
#import matplotlib as mpl
#mpl.use('TkAgg')
#import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import webcolors



def centroid_histogram(clt):
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    # normalize the histogram, such that it sums to one
    hist = hist.astype('float')
    hist /= hist.sum()

    return hist





def color_kmean(imagePath, n_clusters):
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    clt = KMeans(n_clusters = n_clusters)
    clt.fit(image)
    hist = centroid_histogram(clt)
   
    rgb = clt.cluster_centers_
    hsv = cv2.cvtColor(np.uint8([rgb]), cv2.COLOR_RGB2HSV)                                                                     

    return (hist, clt.cluster_centers_, hsv)


## convert rgb score to closet name using Euclidan distance 
def closest_color(requested_color):
    min_colors = {}
    for key, name in webcolors.html4_hex_to_names.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_color[0])**2
        gd = (g_c - requested_color[1])**2
        bd = (b_c - requested_color[2])**2
        min_colors[name] = rd+gd+bd
        
    return sorted(min_colors.items(), key=lambda x:x[1])[0]



def convert_rgb_label(rgb):
    _white = [100, 100, 100]
    _silver = [75,75, 75]
    _gray = [50,50, 50]
    _black = [0, 0, 0]
    _red = [100, 0, 0]
    _maroon = [50, 0, 0]
    _yellow = [100, 100, 0]
    _olive = [50,50,0]
    _lime = [0, 100, 0]
    _green = [0, 50, 0]
    _aqua = [0, 100, 100]
    _teal = [0, 50, 50]
    _blue = [0, 0, 100]
    _navy = [0, 0, 50]
    _fuchsia = [100, 0, 100]
    _purple = [50, 0, 50]
    
    dis = {}
    dis2 = {}
    labels = ['white', 'silver', 'gray', 'black', 'red', 'maroon', \
             'yellow', 'olive', 'lime', 'green', 'aqua', 'teal', \
             'blue', 'navy', 'fuchsia', 'purple']
    codes = [_white, _silver, _gray, _black, _red, _maroon, \
            _yellow, _olive, _lime, _green, _aqua, _teal, \
            _blue, _navy, _fuchsia, _purple]
    
    for (_, code) in zip(labels, codes):
        code_ = tuple(map(lambda x:str(x)+'%', code))
        reference = webcolors.rgb_percent_to_rgb(code_)
        dis[_] = np.linalg.norm(np.array(rgb) - np.array(reference))
        dis2[_] = np.sum(np.abs(np.array(rgb) - np.array(reference)))
        #print(_, rgb, code_, reference, dis[_], dis2[_])
        name = closest_color(rgb)
        
        
    return dis, name


def color_kmean(imagePath, n_clusters):
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    clt = KMeans(n_clusters = n_clusters)
    clt.fit(image)
    hist = centroid_histogram(clt)
   
    rgb = clt.cluster_centers_
    hsv = cv2.cvtColor(np.uint8([rgb]), cv2.COLOR_RGB2HSV)

    return (hist, rgb, hsv)

def predict_color(imagepath, nclusters=5):
    files = os.listdir(imagepath)
    print(os.getcwd())
    imagepath = [os.path.join(imagepath, f) for f in files if f.endswith('.jpg')]
    length = len(imagepath)


    totalprob = {}

    for _ in imagepath:
        hist, rgb, hsv = color_kmean(_, nclusters)
        for i in range(len(rgb)):
            name = closest_color(rgb[i])[0]
            if name not in totalprob:
                totalprob[name] = hist[i]
            else:
                totalprob[name] += hist[i]

    return totalprob


def predict_color(imagepath, nclusters=5):
    files = os.listdir(imagepath)
    print(os.getcwd())
    imagepath = [os.path.join(imagepath, f) for f in files if f.endswith('.jpg')]
    length = len(imagepath)


    totalprob = {}
    totalProb2 = []
    for _ in imagepath:
        #print('image: ', _)
        hist, rgb, hsv = color_kmean(_, nclusters)
        name = [0]*len(hist)

        for i in range(len(rgb)):
            name[i] = closest_color(rgb[i])[0]
            if name[i] not in totalprob:
                totalprob[name[i]] = hist[i]
            else:
                totalprob[name[i]] += hist[i]

        dic = dict(zip(name, hist))
        dic = sorted(dic.items(), key=lambda x: x[1], reverse=True)
        if len(dic) >= 3:
            totalProb2.append([dic[0][0], dic[1][0], dic[2][0]])
        else:
            totalProb2.append([dic[0][0], dic[1][0], 'None'])
        print(dic)

    return totalprob, totalProb2


if __name__ == '__main__':
    #imagePath = '../board/interior/'
    #imagePath = '../../../clothing-pattern-dataset/ColorImage3/black/'
    imagePath = '../../../clothing-pattern-dataset/ColorImage3/yellow/'
    result, result2 = predict_color(imagePath)
    result = sorted(result.items(), key=lambda x: x[1])
    print(result)
    print(result2)
    
