from __future__ import division
import cv2
import numpy as np
import os
import sys
import argparse
import time
from math import exp, pow
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt
import collections

from augmentingPath import augmentingPath
from pushRelabel import pushRelabel
from boykovKolmogorov import boykovKolmogorov

graphCutAlgo = {"ap": augmentingPath, 
                "pr": pushRelabel, 
                "bk": boykovKolmogorov}
SIGMA = 7.0  # smaller means more sensitive to edges, smaller cuts

CUTCOLOR = (0, 0, 255)

SF = 10

def show_image(image):
    windowname = "Segmentation"
    cv2.namedWindow(windowname, cv2.WINDOW_NORMAL)
    cv2.startWindowThread()
    cv2.imshow(windowname, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
# Large when ip - iq < sigma, and small otherwise
def boundaryPenalty(ip, iq):
    bp = 100 * exp(- pow(int(ip) - int(iq), 2) / (2 * pow(SIGMA, 2)))
    return bp

def buildGraph(image: np.ndarray, SOURCE: int, SINK: int):
    graph = collections.defaultdict(dict)
    K = makeNLinks(graph, image)
    makeTLinks(graph, image, K, SOURCE, SINK)
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

def makeTLinks(graph: collections.defaultdict(dict), image: np.ndarray, K: float, SOURCE: int, SINK: int):
    r, c = image.shape

    graph[SINK] = {}  # sink has no children

    for i in range(r):
        for j in range(c):
            x = i * c + j

            # darker pixels weighted towards source, lighter towards sink
            if (image[i,j] < 70):
                graph[SOURCE][x] = 100
                graph[x][SOURCE] = 0
            else:
                graph[SOURCE][x] = 0
                graph[x][SOURCE] = 0

            if (image[i,j] >= 100):
                graph[x][SINK] = 100
                graph[SINK][x] = 0
            else:
                graph[x][SINK] = 0
                graph[SINK][x] = 0

def displayCut(image, cuts, SOURCE, SINK):
    def colorPixel(i, j):
        image[i][j] = CUTCOLOR

    rows, cols = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    for c in cuts:
        if c != SOURCE and c != SINK:
            colorPixel(c // cols, c % cols)
    return image

def plotHistogram(image):

    histogram, bin_edges = np.histogram(image, bins=256, range=(0, 255))

    # configure and draw the histogram figure
    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("grayscale value")
    plt.ylabel("pixel count")
    plt.xlim([0.0, 255.0])  # <- named arguments do not work here

    plt.plot(bin_edges[0:-1], histogram)  # <- or here
    plt.show()

def imageSegmentation(imagefile, size=(30, 30), algo="ff"):
    pathname = os.path.splitext(imagefile)[0]
    image = cv2.imread(imagefile, cv2.IMREAD_GRAYSCALE)
    originalImage = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    originalSize = image.shape
    print("Input image size: {}".format(originalSize))

    # apply median filter to remove noise
    image_filtered = median_filter(image, (10,10))

    resizeFactor = 0.2
    newSize = (int(resizeFactor * image.shape[1]), int(resizeFactor * image.shape[0]))

    image = cv2.resize(image_filtered, newSize)
    print("Resized image: {}".format(image.shape))

    # rescale image intensity to (0, 255)
    min_val = np.min(image)
    image = image - min_val
    max_val = np.max(image)
    image = (image * (255 / max_val)).astype(np.uint8)
    cv2.imwrite(pathname + "_preprocessed.jpg", image)

    #plotHistogram(image)

    SOURCE = (image.shape[0] * image.shape[1])
    SINK   = (image.shape[0] * image.shape[1] + 1)

    graph = buildGraph(image, SOURCE, SINK)
    
    start = time.time()
    cuts = graphCutAlgo[algo](graph, SOURCE, SINK)

    print("Elapsed: {}".format(time.time() - start))
    image = displayCut(image, cuts, SOURCE, SINK)
    image = cv2.resize(image, (originalSize[1], originalSize[0]))
    show_image(image)

    combinedImg = np.hstack((originalImage, image))
    savename = pathname + "before_after.jpg"
    cv2.imwrite(savename, combinedImg)
    print("Saved image as {}".format(savename))

def parseArgs():
    def algorithm(string):
        if string in graphCutAlgo:
            return string
        raise argparse.ArgumentTypeError(
            "Algorithm should be one of the following:", graphCutAlgo.keys())

    parser = argparse.ArgumentParser()
    parser.add_argument("imagefile")
    parser.add_argument("--size", "-s", 
                        default=30, type=int,
                        help="Defaults to 30x30")
    parser.add_argument("--algo", "-a", default="ap", type=algorithm)
    return parser.parse_args()

if __name__ == "__main__":

    args = parseArgs()
    imageSegmentation(args.imagefile, (args.size, args.size), args.algo)
