# 'Efficient Graph-Based Image Segmentation' (Felzenswalb IJCV 2005) in Python
# Original C++ code available at http://cs.brown.edu/~pff/segment/
# author: Muhammet Bastan, mubastan@gmail.com
# date: February 2012

import numpy as np


# import operator


class Edge:
    """
    Graph-theory edge
    """
    def __init__(self, a=0, b=1, w=0):
        self.set(a, b, w)

    def set(self, a, b, w):
        self.a = a
        self.b = b
        self.w = w

    def __lt__(self, other):
        return self.w < other.w


class Node:
    """
    Graph-theory node
    """
    def __init__(self, parent, rank=0):
        self.parent = parent
        self.rank = rank
        self.size = 1


class depthFirstSearch:
    def __init__(self, numElements):
        self.numElements = numElements
        self.numSets = numElements
        self.nodes = np.array([Node(i) for i in range(numElements)])

    def find(self, x):
        y = x
        while y != self.nodes[y].parent:
            y = self.nodes[y].parent
        self.nodes[y].parent = y
        return y

    def union(self, x, y):
        xr = self.find(x)  # root, set of x
        yr = self.find(y)  # root, set of y
        if (x == y or xr == yr):
            return
        nx = self.nodes[xr]
        ny = self.nodes[yr]
        if nx.rank > ny.rank:
            ny.parent = xr
            nx.size += ny.size
        else:
            nx.parent = yr
            ny.size += nx.size
            if nx.rank == ny.rank:
                ny.rank += 1
        self.numSets -= 1

    def setSize(self, id):
        return self.nodes[id].size

    def reset(self):
        for i in range(self.numElements):
            self.nodes[i].rank = 0
            self.nodes[i].size = 1
            self.nodes[i].parent = i
        self.numSets = self.numElements


class eGraphBasedSegment:

    def __init__(self, width, height, threshold=300.0, minSize=10):
        self.W = width      # width of the image
        self.H = height     # height of the image
        self.numEdges = 0  # initialize to 0, will be set according to segmentation type
        self.setParameters(threshold, minSize)
        self.init(width, height)

    def setParameters(self, threshold=300.0, minSize=10):
        self.TH = float(threshold)
        self.MSZ = minSize

    def init(self, width, height):
        size = width*height
        nedges = 2*size - width - height
        self.edges = np.array([Edge() for i in range(nedges)])
        self.dfs = depthFirstSearch(size)
        self.thresholds = self.TH*np.ones(size)

    # number of sets/components in the segmentation
    def numSets(self):
        return self.dfs.numSets

    def segmentImage(self, image):
        self.numEdges = self.buildGraph(image)
        self.segmentGraph(self.numEdges)

    def segmentEdgeImage(self, edgeImage):
        self.numEdges = self.buildEdgeGraph(edgeImage)
        self.segmentGraph(self.numEdges)

    def buildGraph(self, image):
        w = image.shape[1]
        h = image.shape[0]
        numEdges = 0
        yw, ywx = 0, 0
        for y in range(h):
            yw = y*w
            for x in range(w):
                ywx = yw + x
                if x < w - 1:
                    dist = np.sqrt(sum((image[y, x, :] - image[y, x+1, :])**2))
                    self.edges[numEdges].set(ywx, ywx+1, dist)
                    numEdges += 1
                if y < h - 1:
                    dist = np.sqrt(sum((image[y, x, :] - image[y+1, x, :])**2))
                    self.edges[numEdges].set(ywx, ywx + w, dist)
                    numEdges += 1
        return numEdges

    # cmag: magnitude of color gradient, from color canny
    def buildEdgeGraph(self, cmag):
        w = cmag.shape[1]
        h = cmag.shape[0]
        numEdges = 0
        yw, ywx = 0, 0
        for y in range(h):
            yw = y*w
            for x in range(w):
                ywx = yw + x
                if x < w - 1:
                    self.edges[numEdges].set(ywx, ywx+1, cmag[y, x])
                    numEdges += 1
                if y < h - 1:
                    self.edges[numEdges].set(ywx, ywx + w, cmag[y, x])
                    numEdges += 1
        return numEdges

    def segmentGraph(self, numEdges):
        self.edges.sort()
        for i in range(numEdges):

            # self.edges[i].printEdge()
            ed = self.edges[i]
            a = self.dfs.find(ed.a)
            b = self.dfs.find(ed.b)
            if a != b:
                w = ed.w
                if (w <= self.thresholds[a]) and (w <= self.thresholds[b]):
                    # tha = self.thresholds[a]
                    # thb = self.thresholds[b]
                    self.dfs.union(a, b)
                    a = self.dfs.find(a)
                    asize = self.dfs.nodes[a].size
                    self.thresholds[a] = w + float(self.TH)/(asize)
                    # self.thresholds[a] = w + float(self.TH+asize)/(asize*asize)
                    # self.thresholds[a] = w

    # merge small components (post-processing)
    def mergeSmall(self, th=-1, numSegments=-1):
        for i in range(self.numEdges):
            ed = self.edges[i]

            if th > 0 and ed.w > th:
                continue

            a = self.dfs.find(ed.a)
            b = self.dfs.find(ed.b)
            if (a != b) and (self.dfs.nodes[a].size < self.MSZ) or (self.dfs.nodes[b].size < self.MSZ):
                self.dfs.union(a, b)
            if numSegments > 0 and self.dfs.numSets <= numSegments:
                break

    def getLabels(self):
        labels = np.zeros(self.H*self.W)
        cids = []
        for pix in range(self.W*self.H):
            cid = self.dfs.find(pix)
            if cid not in cids:
                cids.append(cid)

                labels[pix] = cids.index(cid)
        labels.shape = (self.H, self.W)
        return labels

    def getSegmentEdges(self):
        labels = self.getLabels()
        edges = np.zeros((self.H, self.W), dtype=bool)
        for y in range(self.H-1):
            for x in range(self.W-1):
                if labels[y, x] != labels[y, x+1] or labels[y, x] != labels[y+1, x]:
                    edges[y, x] = 255

        return labels, edges
