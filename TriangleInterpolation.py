'''
Created on Jul 22, 2019

@author: anshul
'''

import numpy as np
import Gravity
import pandas as pd
import math as m
import mathutils
import DataPrep

class Coordinate:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def distance(p1, p2):
    return m.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def barycentric_weights(P, A, B, C):
    AB = (B.x - A.x, B.y - A.y)
    AC = (C.x - A.x, C.y - A.y)
    AP = (P.x - A.x, P.y - A.y)
    d00 = np.dot(AB, AB)
    d01 = np.dot(AB, AC)
    d11 = np.dot(AC, AC)
    d20 = np.dot(AP, AB)
    d21 = np.dot(AP, AC)

    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return u


def sigmoid_func(x):
    return (1 + 0.5 / (1+ m.exp(-0.26 * (x-18))))

def find_average(poss):
    zone_one = {}
    zone_two = {}
    zone_three = {}
    zone_four = {}
    zone_five = {}
    zone_one['time'] = []
    zone_two['time'] = []
    zone_three['time'] = []
    zone_four['time'] = []
    zone_five['time'] = []
    onball_gravity = {}
    onball_gravity['time'] = []
    onball_gravity['zone'] = []
    onball_gravity['player'] = []
    onball_gravity['gravity'] = []
    
    dict_lst = [zone_one, zone_two, zone_three, zone_four, zone_five]
    for d in dict_lst:
        if dict_lst.index(d) == 0:
            bounds = [0, 3]
        if dict_lst.index(d) == 1:
            bounds = [3, 5]
        if dict_lst.index(d) == 2:
            bounds = [5, 15]
        if dict_lst.index(d) == 3:
            bounds = [15, 20.75]
        if dict_lst.index(d) == 4:
            bounds = [20.75, 100]
        for time, moment in poss.items():
            d['time'].append(time)
            for k, v in moment.items():
                if k not in d:
                    d[k] = []
                if len(v) == 4:
                    defender, offense, ball, hoop = v[0], v[1], v[2], v[3]
                    if bounds[0] <= distance(offense, hoop) < bounds[1]:
                        weightA = barycentric_weights(defender, offense, ball, hoop)
                        point = mathutils.geometry.intersect_point_line((offense.x, offense.y), (hoop.x, hoop.y), (ball.x, ball.y))[0]
                        point = Coordinate(point[0], point[1])
                        while weightA < 0.0:
                            x_diff = defender.x - offense.x
                            y_diff = defender.y - offense.y
                            offense.x += (2*x_diff)
                            offense.y += (2*y_diff)
                            weightA = barycentric_weights(defender, offense, ball, hoop)
                        if weightA > 1:
                            weightA = barycentric_weights(offense, defender, ball, hoop)
                        ball_multiplier = sigmoid_func(distance(defender, ball))
                        hoop_multiplier = sigmoid_func(distance(defender, hoop))
                        weightA = float(weightA) * ball_multiplier * hoop_multiplier
                        if distance(point, offense) < 3:
                            if len(d[k]) > 0:
                                weightA = 1
                            else:
                                weightA = 1
                        if weightA > 2.25:
                            weightA = 2.25
                        d[k].append(weightA)
                    else:
                        d[k].append("N/A")
                if len(v) == 5 and dict_lst.index(d) == 0:
                    onball_gravity['time'].append(time)
                    onball_gravity['player'].append(k)
                    defender, offense, ball, hoop = v[0], v[1], v[2], v[3]
                    dist = distance(offense, hoop)
                    if dist > 0:
                        zone = 1
                    if dist > 3:
                        zone = 2
                    if dist > 5:
                        zone = 3
                    if dist > 15:
                        zone = 4
                    if dist > 20.75:
                        zone = 5
                    onball_gravity['zone'].append(zone)
                    weight = mathutils.geometry.intersect_point_line((defender.x, defender.y), (hoop.x, hoop.y), (offense.x, offense.y))[1]
                    hoop_multiplier = sigmoid_func(distance(defender, hoop))
                    weight_new = float(weight) * hoop_multiplier
                    onball_gravity['gravity'].append(weight_new)
    return [zone_one, zone_two, zone_three, zone_four, zone_five, onball_gravity]
    #for key in d.keys():
        #print(key, len(d[key]))
    #return([(k, d[k]) for k in d.keys() if k != 'time' and len(d[k]) != 0])
    #return(offball_gravity, ball_gravity)
    #for row in df.iterrows():
        #print(row)
    #df.to_csv("poss_test_18.csv")
        
def create_visual(game, time):
    vectorDict = Gravity.get_vectors()[0]
    momentList = vectorDict[game]
    for d in momentList:
        if max(d.keys()) == time:
            poss = d
    d = {}
    d['time'] = []
    d['ball_handler'] = []
    for time, moment in poss.items():
        d['time'].append(time)
        for k, v in moment.items():
            #print("hey there")
            if k not in d:
                d[k] = []
            if len(v) == 4:
                defender, offense, ball, hoop = v[0], v[1], v[2], v[3]
                weightA = barycentric_weights(defender, offense, ball, hoop)
                point = mathutils.geometry.intersect_point_line((offense.x, offense.y), (hoop.x, hoop.y), (ball.x, ball.y))[0]
                point = Coordinate(point[0], point[1])
                if weightA < 0.0:
                    x_diff = defender.x - offense.x
                    y_diff = defender.y - offense.y
                    offense.x += (2*x_diff)
                    offense.y += (2*y_diff)
                    weightA = barycentric_weights(defender, offense, ball, hoop)
                if weightA > 1:
                    weightA = barycentric_weights(offense, defender, ball, hoop)
                ball_multiplier = sigmoid_func(distance(defender, ball))
                hoop_multiplier = sigmoid_func(distance(defender, hoop))
                weightA = float(weightA) * ball_multiplier * hoop_multiplier
                if distance(point, offense) < 3:
                    weightA = d[k][-1][0]
                if weightA > 2.25:
                    weightA = 2.25
                d[k].append([weightA, defender.x, defender.y])
            if len(v) == 5:
                d['ball_handler'].append(k)
                defender, offense, ball, hoop = v[0], v[1], v[2], v[3]
                weight = mathutils.geometry.intersect_point_line((defender.x, defender.y), (hoop.x, hoop.y), (offense.x, offense.y))[1]
                hoop_multiplier = sigmoid_func(distance(defender, hoop))
                weight_new = float(weight) * hoop_multiplier
                d[k].append([weight_new, defender.x, defender.y])
    lst = []
    df = pd.DataFrame(d)
    cols = df.columns
    players = list(cols)
    players.remove('time')
    players.remove('ball_handler')
    players.sort()
    final_cols = ['time']
    final_cols.extend(players)
    df = df.reindex(columns=final_cols)
    
    for row in df.itertuples():
        temp = []
        temp.append(row.time)
        for i in range(2, len(row)):
            temp.extend(row[i])
        lst.append(temp)
    df = pd.DataFrame(lst, columns = ["time", players[0], "p1_x", "p1_y", players[1], "p2_x", "p2_y", players[2], "p3_x", "p3_y", players[3], "p4_x", "p4_y", players[4], "p5_x", "p5_y" ])
    df.to_csv("poss_test_23.csv")
    print(df)

if __name__ == '__main__':
    game = 20150313
    time = 211.8
    #create_visual(game, time)
    #print(sigmoid_func(24))
    vectorDict = Gravity.get_vectors()
    momentList = vectorDict[game]
    for d in momentList:
        if max(d.keys()) == 211.8:
            poss = d
    find_average(poss)
