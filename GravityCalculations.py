# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 10:17:55 2019

@author: schmi
"""
import math as m
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as sp
import Gravity
import scipy.optimize
import pandas as pd


"""vectorDict = Gravity.get_vectors()
gameID = 20141114
time = 51.88
momentList = [49.69,50.65,51.36,51.88,52.79,53.54]"""


class Vector:
    def __init__(self,deltax,deltay):
        self.deltax = deltax
        self.deltay = deltay
        self.dist = m.sqrt(deltax**2 + deltay**2)
        if deltax == 0:
            self.direction = 0
        else:
            self.direction = m.atan(deltay / deltax)
class FCurve:
    def __init__(self, q_1, q_2, k=-10):
        self.k = -100
        self.q_1 = q_1
        self.q_2 = q_2 
        self.curveFunc = self.computeCurveFunc()
        
    def computeCurveFunc(self):
        def curve(r):
#            r[r==0] = 0.1
            if r == 0:
                return 10
            coul =  (self.k * self.q_1 * self.q_2) / (r**2)
            #Hella low value for pauli... like 10 to the negative 9, 10, etc.
            #pauli = -self.C * np.exp(-self.a * r) * (1/r**2 - self.a)
            #print(coul, pauli)
            return coul #+ pauli
        return curve
    
    def setq_1(self,q_1_new):
        self.q_1 = q_1_new
        self.curveFunc = self.computeCurveFunc()
        
    def setq_2(self,q_2_new):
        self.q_2 = q_2_new
        self.curveFunc = self.computeCurveFunc()
        
    def setC(self,C_new):
        self.C = C_new
        self.curveFunc = self.computeCurveFunc()
        
    def seta(self,a_new):
        self.a = a_new
        self.curveFunc = self.computeCurveFunc()
        
    def setk(self,k_new):
        self.k = k_new
        self.curveFunc = self.computeCurveFunc()

def makeVectors(moment):
    #change the dimension to (5,7) once the ball and net get added in
    vecs = np.zeros((5,7),dtype=Vector)
    players = []
    for i,player in enumerate(moment):
        players.append(player)
        for j,coords in enumerate(moment.get(player)):
            vecs[i][j] = Vector(coords[0],coords[1])
    return vecs, players

def simpleSolve(moment):
    vectors = makeVectors(moment)[0]
    inverseSquares = np.zeros(vectors.shape)
    angles = np.zeros(vectors.shape)
    for i in range(vectors.shape[0]):
        for j in range(vectors.shape[1]):
#            print("i = " + str(i) + " j = " + str(j))
#            print(vectors[i][j].dist)
            inverseSquares[i][j] = 1 / (vectors[i][j].dist ** 2)
            angles[i][j] = vectors[i][j].direction
    matrixA = np.zeros(vectors.shape)
    matrixB = np.zeros(vectors.shape)
    for i in range(vectors.shape[0]):
        for j in range(vectors.shape[1]):
            matrixA[i][j] = inverseSquares[i][j] * m.cos(angles[i][j])
            matrixB[i][j] = inverseSquares[i][j] * m.sin(angles[i][j])
    system = np.concatenate((matrixA,matrixB), axis = 0)
#    zeros = np.zeros((10,1))
#    system = np.concatenate((system,zeros),axis = 1)
    def makefunc(system):
        def func(weights):
#            weights = np.concatenate((weights,[1]),axis=0)
            product = np.matmul(system,weights)
            total = 0
            for i in product:
                total = total + i**2
            return total
        return func
    myFunc = makefunc(system)
 #   ans = scipy.optimize.minimize(myFunc,np.ones((6,1)))
    ans = scipy.optimize.minimize(myFunc,np.ones((5,1)))
    print(ans)
    solution = -1 * ans.x / max(-1 * ans.x)
#    solution = np.concatenate((solution,[1]),axis=0)
    return solution
    
def gravityOverTime(momentList):
    gravDict = {}
    for item in momentList:
        moment = vectorDict.get(gameID).get(item)
        sol = twoPartSolve(moment,playerLabels = True)
        for c in np.transpose(sol):
            if c[0] not in gravDict:
                gravDict.update({c[0]:[]})
            gravDict.get(c[0]).append(c[2:])
            #print(c[2:])
    return gravDict

def averageCharges(momentList,
                   absValues = False,
                   printResult = False,
                   computeStddev = False):
    gravDict = gravityOverTime(momentList)
    print(type(gravDict))
    avgDict = {}
    for key in gravDict:
        player = key
        vals = gravDict.get(key)
        mySum = 0
        myCount = 0
        myList = []
        for i in vals:
            if(absValues == True):
                mySum = abs(i[0]) + mySum
                if(computeStddev):
                    myList.append(abs(i[0]))
            else:
                mySum = i[0] + mySum
                if(computeStddev):
                    myList.append(i[0])
            myCount = myCount + 1
        if(computeStddev):
            avgDict.update({player:(mySum/myCount,np.std(myList))})
        avgDict.update({player: (mySum / myCount)})
    if printResult:
        for key in avgDict:
            if(computeStddev):
                print(str(key) + ": " +
                      str(avgDict.get(key)[0]) + ", " + 
                      str(avgDict.get(key)[1]))
            else:
                print(str(key) + ": " + 
                      str(avgDict.get(key)[0]))
    return avgDict

def dissociationEnergy(curve,radius):
    q_1 = curve.q_1
    q_2 = curve.q_2
    C = curve.C
    a = curve.a
    def U(r):
        return C*np.exp(-a * r) / r - q_1 * q_2 / (r**2)
    return U(radius)

def twoPartSolve(moment,playerLabels=True):
    vectors = makeVectors(moment)[0]
    players = np.asarray(makeVectors(moment)[1])
    cosMat = np.zeros(vectors.shape)
    sinMat = np.zeros(vectors.shape)
    dists = np.zeros(vectors.shape)
    for i in range(vectors.shape[0]):
        for j in range(vectors.shape[1]):
#            print("i = " + str(i) + " j = " + str(j))
#            print(vectors[i][j].dist)
            #print(vectors)
            dists[i][j] = vectors[i][j].dist
            cosMat[i][j] = m.cos(vectors[i][j].direction)
            sinMat[i][j] = m.sin(vectors[i][j].direction)
    trigMat = np.concatenate((cosMat,sinMat),axis=0)
#    zeros = np.zeros((10,1))
#    system = np.concatenate((system,zeros),axis = 1)
    def makefunc(dists,trigMat):
        def func(weights):
            weights = np.reshape(weights,(2,5))
            offenseData = weights[1]
            #print(offenseData[0])
            #print(weights)
            defenseData = weights[0]
            #print(defenseData)
            #print("--------------------------------")
            #defenseData = -1 * np.ones(5)
            NET_WEIGHT = -4
            NET_C = 1
            NET_A = 1
            BALL_WEIGHT = -2
            BALL_A = 1
            BALL_C = 1
            curveSet = []
            forces = np.zeros(dists.shape)
            for o in range(len(offenseData)):
                curves = []
                for d in defenseData:
                    curves.append(FCurve(offenseData[o],
                                         d))
                curves.append(FCurve(offenseData[o], NET_WEIGHT))
                curves.append(FCurve(offenseData[o], BALL_WEIGHT))
                curveSet.append(curves)
            total = 0
            for i in range(5):
                for j in range(7):
                    forces[i][j] = curveSet[i][j].curveFunc(dists[i][j])
            forces = np.concatenate((forces,forces),axis=0)
            #print(forces)
            comps = np.multiply(trigMat,forces)
            #print(comps)
            #print("-----------------------------")
            for row in comps:
                total += (sum(row) ** 2)
            return total
        return func
    myFunc = makefunc(dists,trigMat)
    initialMat = np.ones((2,5))
    initialMat[0] = -1 * initialMat[0]
    #print(initialMat)
    bnd = ((1, 3), (1, 3), (0, 3), (0, 3), (0, 3), (-3, 0), (-3, 0), (-3, 0), (-3, 0), (-3, 0) )
    ans = scipy.optimize.minimize(myFunc, initialMat, method = 'L-BFGS-B')
#    print(ans)
    solution = ans.x
    solution = np.reshape(solution,(2,5))
    print(solution)
#    print(solution)
    radii = []
    for count,Def in enumerate(solution[0]):
        rSum = 0
        els = 0
        for off in solution[1]:
            thisCurve = FCurve(Def,off)
            try:
                root = scipy.optimize.root_scalar(thisCurve.curveFunc, bracket = [0.01,1]).root
            except:
                continue
#            if(root > 10):
#                continue
            rSum = rSum + root
            els = els + 1
        if els == 0:
            radii.append(-1)
        else:
            radii.append(rSum / els)
    if(playerLabels == True):
        solution = np.column_stack((players,np.transpose(solution)))
        solution = np.transpose(solution)
    strs = ["Off ID ", "Off. Q  ", "Def. Q  ", "radius "]
#    print(strs[0] + str(players))
#    print(strs[1] + str(solution[0]))
#    print(strs[2] + str(solution[1]))
#    print(strs[3] + str(radii))
    return solution
#
"""
IDEA: Use the weights matrix from the previous moment so the results are consistent
"""
def showCurves(moment):
    players = makeVectors(moment)[1]
    out = twoPartSolve(moment,playerLabels = False)
    formats = ['b-','m-','g-','c-','y-']
    for i in range(out.shape[1]):
        for j in range(out.shape[1]):
            label = '_nolegend_'
            curve = FCurve(out[0][i],out[1][j],out[2][i],out[3][i])
            r = np.linspace(0.2,3,1000)
            vals = curve.curveFunc(r)
            if(j == 4):
                label = str(players[i])
            plt.plot(r,vals,formats[i],label=label)
    plt.plot(np.linspace(0.2,3,1000),np.zeros(1000),'k--')
    plt.ylim([-3,3])
    plt.legend()
    plt.show()

def create_visual(game, time):
    vectorDict = Gravity.get_vectors()
    momentList = vectorDict[game]
    for d in momentList:
        if max(d.keys()) == time:
            poss = d
    lst = []
    for time, moment in poss.items():
        sol = twoPartSolve(moment, playerLabels = True)
        lst.append([time, sol[1][0], sol[1][1], sol[1][2], sol[1][3], sol[1][4]])
    #print(lst)
    df = pd.DataFrame(lst, columns =['Time', sol[0][0], sol[0][1], sol[0][2], sol[0][3], sol[0][4]]) 
    df = df.round(5)
    print(df)
    #df.to_csv("possession_test7.csv")


def analyze(gameID):
    vectorDict = Gravity.get_vectors()
    player_gravity = {}
    for game, timeDict in vectorDict.items():
        if game == gameID:
            for d in timeDict:
                tempDict = {}
                for time, moment in d.items():
                    sol = twoPartSolve(moment, playerLabels = True)
                    for i in range(5):
                        if sol[0][i] not in player_gravity:
                            player_gravity[sol[0][i]] = []
                        if sol[0][i] not in tempDict:
                            tempDict[sol[0][i]] = []
                        if -3 < sol[2][i] < 3:
                            tempDict[sol[0][i]].append(sol[2][i])
                for k, v in tempDict.items():
                    if len(v) != 0:
                        player_gravity[k].append(sum(v)/len(v))

    for k, v in player_gravity.items():
        print(k, (sum(v)/ len(v)))
#
#def makeSystem(posVectors):
#    system = np.zeros((10,14))
#    for i in range(10):
#        for j,v in enumerate(posVectors):
#            mag = forceMag(v.dist)
#            ang = v.direction
#            if(i < 5):
#                Fx = mag * m.cos(ang)
#                system[i][j] = Fx
#            else:
#                Fy = mag * m.sin(ang)
#                system[i][j] = Fy
#    solution = np.zeros(10)
#    Qs = np.linspace(0,10,100)
#    return

if __name__ == '__main__':
    create_visual(20141115, 494.93)
    #analyze(20141115)
