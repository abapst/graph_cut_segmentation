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

from augmentingPath import augmentingPath
from pushRelabel import pushRelabel
from boykovKolmogorov import boykovKolmogorov

graphCutAlgo = {"ap": augmentingPath, 
                "pr": pushRelabel, 
                "bk": boykovKolmogorov}
SIGMA = 7.0  # smaller means more sensitive to edges, smaller cuts
OBJCOLOR, BKGCOLOR = (0, 0, 255), (0, 255, 0)
OBJCODE, BKGCODE = 1, 2
OBJ, BKG = "OBJ", "BKG"

CUTCOLOR = (0, 0, 255)

SOURCE, SINK = -2, -1
SF = 10

def show_image(image):
    windowname = "Segmentation"
    cv2.namedWindow(windowname, cv2.WINDOW_NORMAL)
    cv2.startWindowThread()
    cv2.imshow(windowname, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def plantSeed(image):

    def drawLines(x, y, pixelType):
        if pixelType == OBJ:
            color, code = OBJCOLOR, OBJCODE
        else:
            color, code = BKGCOLOR, BKGCODE
        cv2.circle(image, (x, y), radius, color, thickness)
        cv2.circle(seeds, (x // SF, y // SF), radius // SF, code, thickness)

    def onMouse(event, x, y, flags, pixelType):
        global drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            drawLines(x, y, pixelType)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            drawLines(x, y, pixelType)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False

    def paintSeeds(pixelType):
        print("Planting {} seeds".format(pixelType))
        global drawing
        drawing = False
        windowname = "Plant " + pixelType + " seeds"
        cv2.namedWindow(windowname, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(windowname, onMouse, pixelType)
        while (1):
            cv2.imshow(windowname, image)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cv2.destroyAllWindows()
    
    seeds = np.zeros(image.shape, dtype="uint8")
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image = cv2.resize(image, (0, 0), fx=SF, fy=SF)

    radius = 10
    thickness = -1 # fill the whole circle
    global drawing
    drawing = False

    #paintSeeds(OBJ)
    #paintSeeds(BKG)
    return seeds, image

# Large when ip - iq < sigma, and small otherwise
def boundaryPenalty(ip, iq):
    bp = 100 * exp(- pow(int(ip) - int(iq), 2) / (2 * pow(SIGMA, 2)))
    return bp

def buildGraph(image):
    V = image.size + 2
    graph = np.zeros((V, V), dtype='int32')
    K = makeNLinks(graph, image)
    seeds, seededImage = plantSeed(image)
    makeTLinks(graph, image, seeds, K)
    return graph, seededImage

def makeNLinks(graph, image):
    K = -float("inf")
    r, c = image.shape
    for i in range(r):
        for j in range(c):
            x = i * c + j
            if i + 1 < r: # pixel below
                y = (i + 1) * c + j
                bp = boundaryPenalty(image[i][j], image[i + 1][j])
                graph[x][y] = graph[y][x] = bp
                K = max(K, bp)
            if j + 1 < c: # pixel to the right
                y = i * c + j + 1
                bp = boundaryPenalty(image[i][j], image[i][j + 1])
                graph[x][y] = graph[y][x] = bp
                K = max(K, bp)
    return K

def makeTLinks(graph, image, seeds, K):
    r, c = seeds.shape

    for i in range(r):
        for j in range(c):
            x = i * c + j

            # darker pixels weighted towards source, lighter towards sink
            #prob = np.exp(-float(image[i,j]) / 20)
            #graph[SOURCE][x] = 100 * prob
            #graph[x][SINK] = 100 * (1 - prob)

            if (image[i,j] < 70):
                graph[SOURCE][x] = 1
            if (image[i,j] >= 100):
                graph[x][SINK] = 1

            #if seeds[i][j] == OBJCODE:
            #    graph[SOURCE][x] = K
            #elif seeds[i][j] == BKGCODE:
            #    graph[x][SINK] = K

def displayCut(image, cuts):
    def colorPixel(i, j):
        image[i][j] = CUTCOLOR

    rows, cols = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    for c in cuts:
        if c[0] != SOURCE and c[0] != SINK and c[1] != SOURCE and c[1] != SINK:
            colorPixel(c[0] // cols, c[0] % cols)
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
    #image = cv2.equalizeHist(image)

    plotHistogram(image)

    graph, seededImage = buildGraph(image)
    cv2.imwrite(pathname + "seeded.jpg", seededImage)

    global SOURCE, SINK
    SOURCE += len(graph) 
    SINK   += len(graph)
    
    start = time.time()
    cuts = graphCutAlgo[algo](graph, SOURCE, SINK)

    print("Elapsed: {}".format(time.time() - start))
    image = displayCut(image, cuts)
    image = cv2.resize(image, (originalSize[1], originalSize[0]))
    show_image(image)
    savename = pathname + "cut.jpg"
    cv2.imwrite(savename, image)
    print("Saved image as {}".format(savename))

    combinedImg = np.hstack((originalImage, image))
    cv2.imwrite(pathname + "before_after.jpg", combinedImg)


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
