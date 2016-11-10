import collections

############################################################
# Problem 3a

def computeMaxWordLength(text):
    """
    Given a string |text|, return the longest word in |text|.  If there are
    ties, choose the word that comes latest in the alphabet.
    A word is defined by a maximal sequence of characters without whitespaces.
    You might find max() and list comprehensions handy here.
    """
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    return sorted(text.split(" "), key = lambda str: (len(str), str), reverse = True)[0]
    # END_YOUR_CODE

############################################################
# Problem 3b

def manhattanDistance(loc1, loc2):
    """
    Return the Manhattan distance between two locations, where the locations
    are pairs of numbers (e.g., (3, 5)).
    """
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    return abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])
    # END_YOUR_CODE

############################################################
# Problem 3c

def mutateSentences(sentence):
    """
    High-level idea: generate sentences similar to a given sentence.
    Given a sentence (sequence of words), return a list of all possible
    alternative sentences of the same length, where each pair of adjacent words
    also occurs in the original sentence. (The words within each pair should appear
    in the same order in the output sentence as they did in the orignal sentence.)
    Notes:
    - The order of the sentences you output doesn't matter.
    - You must not output duplicates.
    - Your generated sentence can use a word in the original sentence more than
      once.
    """
    # BEGIN_YOUR_CODE (our solution is 20 lines of code, but don't worry if you deviate from this)
    words = sentence.split(" ")
    pairs = {}
    for i in xrange(len(words) - 1):
        if words[i] in pairs.keys() and words[i + 1] not in pairs[words[i]]:
            pairs[words[i]].add(words[i + 1])
        if words[i] not in pairs.keys():
            pairs[words[i]] = set([words[i+1]])

    def subsentences(starter, num):
        if starter not in pairs.keys():
            return None
        if num == 2:
            return [starter + " " + k for k in pairs[starter]]
        else:
            possibles = []
            for nextone in pairs[starter]:
                addons = subsentences(nextone, num - 1)
                if addons != None:
                    for addon in addons:
                        possibles.append(starter + " " + addon)
            return possibles

    allsentences = []
    for starter in pairs.keys():
        allsentences += subsentences(starter, len(words))

    return allsentences
    # END_YOUR_CODE

############################################################
# Problem 3d

def sparseVectorDotProduct(v1, v2):
    """
    Given two sparse vectors |v1| and |v2|, each represented as collection.defaultdict(float), return
    their dot product.
    You might find it useful to use sum() and a list comprehension.
    This function will be useful later for linear classifiers.
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    allkeys = set(v1.keys()).union(set(v2.keys()))
    list1 = [v1[k] if k in v1.keys() else 0 for k in allkeys]
    list2 = [v2[k] if k in v2.keys() else 0 for k in allkeys]
    return sum(list1[k] * list2[k] for k in range(len(allkeys)))
    # END_YOUR_CODE

############################################################
# Problem 3e

def incrementSparseVector(v1, scale, v2):
    """
    Given two sparse vectors |v1| and |v2|, perform v1 += scale * v2.
    This function will be useful later for linear classifiers.
    """
    # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
    allkeys = set(v1.keys()).union(set(v2.keys()))
    for k in allkeys:
        if k in v2.keys() and k in v1.keys():
            v1[k] = v1[k] + scale * v2[k]
        elif k in v2.keys() and not k in v1.keys():
            v1[k] = scale * v2[k]
    # END_YOUR_CODE

############################################################
# Problem 3f

def computeMostFrequentWord(text):
    """
    Splits the string |text| by whitespace and returns two things as a pair:
        the set of words that occur the maximum number of times, and
    their count, i.e.
    (set of words that occur the most number of times, that maximum number/count)
    You might find it useful to use collections.defaultdict(float).
    """
    # BEGIN_YOUR_CODE (our solution is 5 lines of code, but don't worry if you deviate from this)
    times = collections.defaultdict(int)
    for word in text.split(" "):
        if word in times.keys():
            times[word] += 1
        else:
            times[word] = 1
    maxtime = max(times.values())
    words = set()
    for k in times.keys():
        if times[k] == maxtime:
            words.add(k)
    return (words, maxtime)
    # END_YOUR_CODE

############################################################
# Problem 3g

def computeLongestPalindrome(text):
    """
    A palindrome is a string that is equal to its reverse (e.g., 'ana').
    Compute the length of the longest palindrome that can be obtained by deleting
    letters from |text|.
    For example: the longest palindrome in 'animal' is 'ama'.
    Your algorithm should run in O(len(text)^2) time.
    You should first define a recurrence before you start coding.
    """
    # BEGIN_YOUR_CODE (our solution is 19 lines of code, but don't worry if you deviate from this)
    n = len(text)
    if n == 0:
        return 0
    table = [[0 for i in range(n)] for i in range(n)]
    for i in range(n):
        table[i][i] = 1

    sublength = 2
    while sublength <= n:
        for j in range(n - sublength + 1):
            k = j + sublength - 1
            if text[j] == text[k]:
                if sublength == 2:
                    table[j][k] = 2
                else:
                    table[j][k] = table[j + 1][k - 1] + 2
            else:
                table[j][k] = max(table[j + 1][k], table[j][k - 1])
        sublength += 1
    return table[0][n - 1]
    # END_YOUR_CODE
