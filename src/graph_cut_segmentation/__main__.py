from __future__ import division
import cv2
import numpy as np
import os
import sys
import argparse
import time
from math import exp, pow
from scipy.ndimage import median_filter
from scipy.stats import norm
import matplotlib.pyplot as plt
import collections
from sklearn.mixture import GaussianMixture

from graph_cut_segmentation import edmondsKarp
from graph_cut_segmentation import boykovKolmogorov

graphCutAlgo = {"ek": edmondsKarp,
                "bk": boykovKolmogorov}

SIGMA = 7.0  # smaller means more sensitive to edges, smaller cuts

CUTCOLOR = (0, 255, 0)

def showImage(image):
    windowname = "Segmentation"
    cv2.imshow(windowname, image)
    cv2.waitKey(0)
    
# Large when ip - iq < sigma, and small otherwise
def boundaryPenalty(ip, iq):
    bp = int(100 * exp(- pow(int(ip) - int(iq), 2) / (2 * pow(SIGMA, 2))))
    return bp

def buildGraph(image: np.ndarray, SOURCE: int, SINK: int, thresh: int):
    graph = collections.defaultdict(dict)
    K = makeNLinks(graph, image)
    makeTLinks(graph, image, K, SOURCE, SINK, thresh)
    return graph

def makeNLinks(graph: collections.defaultdict(dict), image: np.ndarray) -> float:
    K = -float("inf")
    r, c = image.shape
    for i in range(r):
        for j in range(c):
            x = i * c + j
            if i + 1 < r: # pixel below
                y = (i + 1) * c + j
                bp = boundaryPenalty(image[i][j], image[i + 1][j])

                graph[x][y] = bp
                graph[y][x] = bp
                K = max(K, bp)

            if j + 1 < c: # pixel to the right
                y = i * c + j + 1
                bp = boundaryPenalty(image[i][j], image[i][j + 1])

                graph[x][y] = bp
                graph[y][x] = bp
                K = max(K, bp)
    return K

def makeTLinks(graph: collections.defaultdict(dict), image: np.ndarray, K: float, SOURCE: int, SINK: int, thresh: int):
    r, c = image.shape

    graph[SINK] = {}  # sink has no children

    for i in range(r):
        for j in range(c):
            x = i * c + j

            # darker pixels weighted towards source, lighter towards sink
            if (image[i,j] < thresh):
                graph[SOURCE][x] = 100
                graph[x][SOURCE] = 0
            else:
                graph[SOURCE][x] = 0
                graph[x][SOURCE] = 0

            if (image[i,j] >= thresh):
                graph[x][SINK] = 100
                graph[SINK][x] = 0
            else:
                graph[x][SINK] = 0
                graph[SINK][x] = 0

def displayCut(image, cuts, SOURCE, SINK):

    rows, cols = image.shape
    imageRGB = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    for c in cuts:
        if c != SOURCE and c != SINK:
            imageRGB[c // cols][ c % cols] = CUTCOLOR
    return imageRGB

def displayCutColorMap(image, cuts, SOURCE, SINK):

    cmap = plt.cm.get_cmap('plasma')

    rows, cols = image.shape
    imageRGB = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    for c in cuts:
        if c != SOURCE and c != SINK:
            originalColor = image[c // cols][c % cols] / 255.
            newColor = 255 * np.array(cmap(originalColor))[:3]
            imageRGB[c // cols][c % cols] = newColor
    return imageRGB

def analyzeHistogram(image):

    # configure and draw the histogram figure
    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("grayscale value")
    plt.ylabel("pixel count")
    plt.xlim([0.0, 255.0])  # <- named arguments do not work here

    histogram, bin_edges = np.histogram(image, bins=256, range=(0, 255))
    histogram = np.array(histogram) / np.sum(histogram)
    plt.plot(bin_edges[:-1], histogram)  # <- or here

    gm = GaussianMixture(n_components=2).fit(image.flatten().reshape(-1, 1))
    x_samples = np.arange(0, 255, 0.5)
    y_component1 = gm.weights_.flatten()[0] * norm.pdf(x_samples, gm.means_.flatten()[0], gm.covariances_.flatten()[0] ** 0.5)
    y_component2 = gm.weights_.flatten()[1] * norm.pdf(x_samples, gm.means_.flatten()[1], gm.covariances_.flatten()[1] ** 0.5)

    plt.plot(x_samples, y_component1)
    plt.plot(x_samples, y_component2)

    print("Component means: {}".format(gm.means_.flatten()))

    # find the crossover point between the two peaks of the normal distributions
    # (they may have crossovers to the left or right of the peaks as well)
    mean_loc = [np.argmax(y_component1), np.argmax(y_component2)]
    crossover = x_samples[np.argmin(abs(y_component1[min(mean_loc):max(mean_loc)] - y_component2[min(mean_loc):max(mean_loc)])) + min(mean_loc)]
    print(f"Detected threshold: {crossover}")

    plt.show()
    return crossover

def main(imagefile, resizeFactor: float = 1.0, algo: str = "bk"):
    pathname = os.path.splitext(imagefile)[0]
    image = cv2.imread(imagefile, cv2.IMREAD_GRAYSCALE)
    originalImage = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    originalSize = image.shape
    print("Input image size: {}".format(originalSize))

    # apply median filter to remove noise
    start = time.time()
    image_filtered = median_filter(image, (7, 7))
    print("Median filter time: {}".format(time.time() - start))

    # analyze image histogram to select threshold
    start = time.time()
    thresh = analyzeHistogram(image)
    print("Threshold selection time: {}".format(time.time() - start))

    # this is optional, can reduce image size to speed up compute
    newSize = (int(resizeFactor * image.shape[1]), int(resizeFactor * image.shape[0]))
    image = cv2.resize(image_filtered, newSize)
    print("Resized image: {}".format(image.shape))

    cv2.imwrite(pathname + "_preprocessed.jpg", image)

    SOURCE = (image.shape[0] * image.shape[1])
    SINK   = (image.shape[0] * image.shape[1] + 1)

    start = time.time()
    graph = buildGraph(image, SOURCE, SINK, thresh)
    print("Graph building time: {}".format(time.time() - start))
    
    start = time.time()
    cuts = graphCutAlgo[algo](graph, SOURCE, SINK)
    print("Segmentation time: {}".format(time.time() - start))

    imageRGB = displayCut(image, cuts, SOURCE, SINK)
    imageRGB = cv2.resize(imageRGB, (originalSize[1], originalSize[0]))

    imageColorMap = displayCutColorMap(image, cuts, SOURCE, SINK)
    imageColorMap = cv2.resize(imageColorMap, (originalSize[1], originalSize[0]))

    combinedImg = np.hstack((originalImage, imageRGB, imageColorMap))
    savename = pathname + "_before_after.png"
    cv2.imwrite(savename, combinedImg)
    print("Saved results image as {}".format(savename))
    showImage(combinedImg)

def parseArgs():
    def algorithm(string):
        if string in graphCutAlgo:
            return string
        raise argparse.ArgumentTypeError(
            "Algorithm should be one of the following:", graphCutAlgo.keys())

    parser = argparse.ArgumentParser()
    parser.add_argument("imagefile")
    parser.add_argument("--size", "-s", 
                        default=1.0, type=float,
                        help="Image scale factor, defaults to 1.0")
    parser.add_argument("--algo", "-a", default="bk", type=algorithm)
    return parser.parse_args()

if __name__ == "__main__":
    args = parseArgs()
    main(args.imagefile, args.size, args.algo)
