#!/usr/bin/python

import random
import collections
import math
import sys
from collections import Counter
from util import *

############################################################
# Problem 3: binary classification
############################################################

############################################################
# Problem 3a: feature extraction

def extractWordFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x:
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    words = x.split(" ")
    return collections.Counter(words)
    # END_YOUR_CODE

############################################################
# Problem 3b: stochastic gradient descent

def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note: only use the trainExamples for training!
    You should call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    '''
    weights = {}  # feature => weight
    # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
    def predictor(x):
        if dotProduct(featureExtractor(x), weights) >= 0:
            return 1
        else:
            return -1
    for i in xrange(numIters):
        for example in trainExamples:
            feature = featureExtractor(example[0])
            truth = example[1]
            if truth * dotProduct(feature, weights) < 1:
                increment(weights, eta * truth, feature)
        print "Iteration %d: Training Error %f, Testing Error %f." %(i + 1,
        evaluatePredictor(trainExamples, predictor), evaluatePredictor(testExamples, predictor))
    # END_YOUR_CODE
    return weights

############################################################
# Problem 3c: generate test case

def generateDataset(numExamples, weights):
    '''
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    '''
    random.seed(42)
    # Return a single example (phi(x), y).
    # phi(x) should be a dict whose keys are a subset of the keys in weights
    # and values can be anything (randomize!) with a nonzero score under the given weight vector.
    # y should be 1 or -1 as classified by the weight vector.
    def generateExample():
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        phi = {random.choice(weights.keys()): random.random()}
        if dotProduct(weights, phi) >= 0:
            y = 1
        else:
            y = -1
        # END_YOUR_CODE
        return (phi, y)
    return [generateExample() for _ in range(numExamples)]

############################################################
# Problem 3e: character features

def extractCharacterFeatures(n):
    '''
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    '''
    def extract(x):
        # BEGIN_YOUR_CODE (our solution is 6 lines of code, but don't worry if you deviate from this)
        tightx = x.replace(' ', '').replace('\t', '')
        ngrams = [tightx[i : i+n] for i in xrange(len(tightx) - n + 1)]
        return collections.Counter(ngrams)
        # END_YOUR_CODE
    return extract

############################################################
# Problem 4: k-means
############################################################


def kmeans(examples, K, maxIters):
    '''
    examples: list of examples, each example is a string-to-double dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxIters: maximum number of iterations to run for (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments, (i.e. if examples[i] belongs to centers[j], then assignments[i] = j)
            final reconstruction loss)
    '''
    # BEGIN_YOUR_CODE (our solution is 32 lines of code, but don't worry if you deviate from this)
    n = len(examples)
    assignments = [None] * n
    assignmentsNew = random.sample(range(K) * n, n)
    centers = [dict() for i in range(K)]
    losses = [0] * n
    # Pre-compute the squares of examples to improve efficiency
    exampleSquares = [dotProduct(example, example) for example in examples]
    Iter = 0
    while Iter < maxIters and assignments != assignmentsNew:
        assignments = [assignmentsNew[i] for i in range(n)]
        # Update the centers based ont he new assginments
        groups = collections.Counter(assignments)
        centers = [dict() for i in range(K)]
        for j in xrange(n):
            increment(centers[assignments[j]], 1.0 / groups[assignments[j]], examples[j])
        # Update the assginmetsNew based on new centers
        # Pre-compute the suqares osf new centers to improve efficeincy
        centerSquares = [dotProduct(center, center) for center in centers]
        def distance(i, j):
            return math.sqrt(exampleSquares[i] + centerSquares[j] - 2 * dotProduct(examples[i], centers[j]))
        # Define the fucntion of calculating distances
        for i in xrange(n):
            example = examples[i]
            ds = [distance(i, j) for j in xrange(K)]
            pairs = min((v,i) for i,v in enumerate(ds))
            assignmentsNew[i] = pairs[1]
            losses[i] = pairs[0]
        Iter += 1
    return centers, assignments, sum(losses)
    # END_YOUR_CODE
