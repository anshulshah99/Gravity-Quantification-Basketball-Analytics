# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 12:34:57 2019

@author: schmi
"""
import Gravity as g
import math
# Dict with tuples of (defense id, offense id) as keys and distances between each pair
# of players as values. The id scheme must be consistent; if no ID list is passed to 
# pairUp() the tuples must contain values from 0-4.
pairDistDict = {}

# offList and defList should contain player IDs if passed in, if not it will default
# to using 0-4 as a numbering system for both teams.

def distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def get_pairings(duke_location, opponent_location):
    pairDistDict = {}
    for i in range(5):
        for k in range(5):
            pairDistDict[(i, k)] = distance(opponent_location[i], duke_location[k])
    pairs = []
    offPlayers = [0, 1, 2, 3, 4]
    defPlayers = [0, 1, 2, 3, 4]
    while(len(offPlayers) > 0): 
        currMin = 100
        defMin = -1
        offMin = -1
        for i in defPlayers:
            for j in offPlayers:
                if(pairDistDict.get((i,j)) < currMin):
                    currMin = pairDistDict.get((i,j))
                    defMin = i
                    offMin = j
        if(offMin == -1 or defMin == -1):
            raise RuntimeError('Something went wrong in trying to find pairs.' 
                               + '\n Check that pairDistDict is initialized properly.')
        pairs.append((defMin,offMin))
        defPlayers.remove(defMin)
        offPlayers.remove(offMin)
    #ret = [(opponent_location[a], duke_location[b]) for (a, b) in pairs]
    return pairs, pairDistDict

def swap(pair1, pair2):
    newPair1 = (pair1[0],pair2[1])
    newPair2 = (pair2[0],pair1[1])
    return newPair1, newPair2

def checkSwaps(pair1,pair2, d):
    currSum = d.get(pair1) + d.get(pair2)
    swap1, swap2 = swap(pair1,pair2)
    swapSum = d.get(swap1) + d.get(swap2)
    if(swapSum < currSum):
        return swap1,swap2
    else:
        return pair1,pair2

def newPairing(duke_location, opponents):
    pairs, d = get_pairings(duke_location, opponents)
    myCopy = pairs.copy()
    for i in range(len(pairs)):
        for j in range(len(pairs)):
            if i == j:
                continue
            myCopy[i],myCopy[j] = checkSwaps(myCopy[i],myCopy[j], d)
    return myCopy            
