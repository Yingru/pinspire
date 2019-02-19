import numpy as np
import cv2
from sklearn.cluster import KMeans
import os
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import webcolors


# a k-mean clustering for identify the colors
class ColorCluster(object):
    def __init__(self, n_clusters = 5):
        self.n_clusters = n_clusters

    def _centroid_histogram(self, clt):
        '''
        normalized clt color histogram
        '''
        numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
        (hist, _) = np.histogram(clt.labels_, bins=numLabels)

        # normalize the histogram, such that it sums to one
        hist = hist.astype('float')
        hist /= hist.sum()

        return hist 

    def _closest_color(self, requested_color):
        '''
        convert the rgb score the closest color name using Euclidan distance
        '''
        min_colors = {}
        for key, name in webcolors.html4_hex_to_names.items():
            r_c, g_c, b_c = webcolors.hex_to_rgb(key)
            rd = (r_c - requested_color[0])**2
            gd = (g_c - requested_color[1])**2
            bd = (b_c - requested_color[2])**2

            min_colors[name] = rd+gd+bd
        return sorted(min_colors.items(), key=lambda x: x[1])[0]
            

    def predict(self, imagePath, totalProb={}):
        '''
        predict the color of the image by k-mean clustering
        Input -- imagePath
        Output -- clustered centroid color (converted to html color scheme) 
        '''
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.reshape((image.shape[0] * image.shape[1], 3))

        clt = KMeans(n_clusters = self.n_clusters)
        clt.fit(image)
        hist = self._centroid_histogram(clt)

        self.rgb = clt.cluster_centers_
        self.hsv = cv2.cvtColor(np.uint8([self.rgb]), cv2.COLOR_RGB2HSV)

        for i in range(len(self.rgb)):
            name = self._closest_color(self.rgb[i])[0]

            if name not in totalProb:
                totalProb[name] = hist[i]
            else:
                totalProb[name] += hist[i]

        return totalProb
            
            
if __name__ == '__main__':
    #imagePath = '../../../clothing-pattern-dataset/ColorImage3/black/'
    imagePath = '../../../clothing-pattern-dataset/ColorImage3/yellow/'
    files = os.listdir(imagePath)
    imagePath = [os.path.join(imagePath, f) for f in files]
    classifier = ColorCluster(n_clusters=5) 
    for i in imagePath:
        print(i, classifier.predict(i, {}))
