'''
Created on Jul 2, 2019

@author: anshul
'''
import pandas as pd
import operator
import math as m
import numpy as np
import itertools
import TriangleInterpolation
import BarycentricPairing


class Coordinate:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def check_halfcourt(y_coords, side):
    if side == "above":
        return(all(x > 47 for x in y_coords))
    else:
        return(all(x < 47 for x in y_coords))

def distance(coord_1, coord_2):
    #calculate distance between two points
    dist = m.sqrt((coord_1.x - coord_2.x)**2 + (coord_1.y - coord_2.y)**2)
    return dist

def closest_defenders(duke_player, defense_list):
    ret = []
    for defender in defense_list:
        dist = distance(duke_player, defender)
        ret.append((dist, defender))
    ret = sorted(ret, key = operator.itemgetter(0))
    return(ret[0][1])

  

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
    return pairs


def get_sides():
    sideDict = {}
    df = pd.read_csv("data/all_games.csv")
    df.columns = df.columns.str.replace('.', '_')
    half = 1
    for row in df.itertuples():
        if row.home == 'yes' and row.event_id == 3 and row.half == half:
            half += 1
            if row.game_id not in sideDict:
                sideDict[row.game_id] = []
            duke_location = [Coordinate(row.p1_x, row.p1_y), Coordinate(row.p2_x, row.p2_y), Coordinate(row.p3_x, row.p3_y), Coordinate(row.p4_x, row.p4_y), Coordinate(row.p5_x, row.p5_y)]
            y_coords = [coord.y for coord in duke_location]
            if(all(y > 47 for y in y_coords)):
                sideDict[row.game_id].append("above")
            else:
                sideDict[row.game_id].append("below")
            if len(sideDict[row.game_id]) == 2:
                half = 1
    sideDict[20141126][0] = 'above'
    return(sideDict)
    
def process(gravity_dicts):
    playerDict = {}
    for i in range(len(gravity_dicts) - 1):
        d = gravity_dicts[i]
        for k, v in d.items():
            if k != 'time' and k != 'ball_handler':
                if k not in playerDict:
                    playerDict[k] = [[] for j in range(10)]
                scores = [val for val in v if val != "N/A"]
                if len(scores) > 0:
                    playerDict[k][i] = scores
    #print(playerDict)
    ball_gravity = gravity_dicts[-1]
    for j in range(len(ball_gravity['time'])):
        player = ball_gravity['player'][j]
        ind = ball_gravity['zone'][j] + 4
        grav = ball_gravity['gravity'][j]
        playerDict[player][ind].append(grav)
    return playerDict
            
    
def get_vectors():
    gravityDict = {}
    playerDict = {}
    shots = {}
    possession = False
    offReb = False
    sideDict = get_sides()
    df = pd.read_csv("data/all_games.csv")
    df.columns = df.columns.str.replace('.', '_')
    for row in df.itertuples():
        if row.game_id not in [20150125, 20150117]:
            half = row.half
            time = round(row.game_clock + (1200 * (2 - half)), 2)
            gameID = row.game_id
            side = sideDict[gameID][half-1]
            duke_players = [row.p1_global_id, row.p2_global_id, row.p3_global_id, row.p4_global_id, row.p5_global_id]
            duke_location = [Coordinate(row.p1_x, row.p1_y), Coordinate(row.p2_x, row.p2_y), Coordinate(row.p3_x, row.p3_y), Coordinate(row.p4_x, row.p4_y), Coordinate(row.p5_x, row.p5_y)]
            opponent_location = [Coordinate(row.p6_x, row.p6_y), Coordinate(row.p7_x, row.p7_y), Coordinate(row.p8_x, row.p8_y), Coordinate(row.p9_x, row.p9_y), Coordinate(row.p10_x, row.p10_y)]
            y_coords = [coord.y for coord in duke_location]
            ball = Coordinate(row.ball_x, row.ball_y)
            if row.event_id == 5 and row.home == 'yes':
                offReb = True
            if gameID not in gravityDict:
                print(gameID)
                gravityDict[gameID] = []
            if gameID not in playerDict:
                playerDict[gameID] = []
            if row.event_id == 8 and not possession:
                continue
            if check_halfcourt(y_coords, side) and row.home == 'yes' and not possession and not offReb and row.event_id not in [1, 2, 8, 11, 28]:
                #offReb = False
                possession = True
                possDict = {}
                if side == 'above':
                    hoop = Coordinate(25, 88.75)
                if side == 'below':
                    hoop = Coordinate(25, 5.25)
                if time not in possDict:
                    possDict[time] = {}
            if offReb and row.home == "no":
                offReb = False
            if possession:
                
                if row.home == 'yes':
                    ball_handler = duke_players[row.p_poss - 1]
                    
                if row.time == 1417040249830:
                    possDict = {}
                    possession = False
                    continue
                
                
                if row.event_id in [1, 2]:
                    continue
                
                
                if row.event_id == 8 and row.home == 'no':
                    #gravityDict[gameID].append(possDict)
                    possession = False
                    continue
                
                
                if row.event_id == 3 or row.event_id == 4:
                    if time not in possDict:
                        possDict[time] = {}
                    pairs = BarycentricPairing.newPairing(duke_location, opponent_location)
                    for d, o in pairs:
                        possDict[time][duke_players[o]] = [opponent_location[d], duke_location[o], ball, hoop]
                    possDict[time][ball_handler].append(0)
                    one, two, three, four, five, ball_gravity = TriangleInterpolation.find_average(possDict)
                    final_dict = process([one, two, three, four, five, ball_gravity])
                    playerDict[gameID].append(final_dict)
                    """for (player, gravity_score) in offball:
                        if player not in playerDict[gameID]:
                            playerDict[gameID][player] = ([], []) 
                        playerDict[gameID][player][0].append(gravity_score)
                    for (player, gravity_score) in onball:
                        if player not in playerDict[gameID]:
                            playerDict[gameID][player] = ([], []) 
                        playerDict[gameID][player][1].append(gravity_score)"""

                    if gameID not in shots:
                        shots[gameID] = {}
                    if time not in shots[gameID]:
                        shots[gameID][time] = None
                    shots[gameID][time] = (final_dict, duke_players)
                    possession = False
                    gravityDict[gameID].append(possDict)
                    continue
                
                
                if row.event_id == 5:
                    
                    """if time not in possDict:
                        possDict[time] = {}
                    offball, onball = TriangleInterpolation.find_average(possDict)
                    pairs = BarycentricPairing.newPairing(duke_location, opponent_location)
                    for d, o in pairs:
                        possDict[time][duke_players[o]] = [opponent_location[d], duke_location[o], ball, hoop]
                    for (player, gravity_score) in offball:
                        if player not in playerDict[gameID]:
                            playerDict[gameID][player] = ([], []) 
                        playerDict[gameID][player][0].append(gravity_score)
                    for (player, gravity_score) in onball:
                        if player not in playerDict[gameID]:
                            playerDict[gameID][player] = ([], []) 
                        playerDict[gameID][player][1].append(gravity_score)"""
                    gravityDict[gameID].append(possDict)
                    offReb = True
                    possession = False
                    continue
                
                
                if row.event_id == 6:
                    """if time not in possDict:
                        possDict[time] = {}
                    offball, onball = TriangleInterpolation.find_average(possDict)
                    pairs = BarycentricPairing.newPairing(duke_location, opponent_location)
                    for d, o in pairs:
                        possDict[time][duke_players[o]] = [opponent_location[d], duke_location[o], ball, hoop]
                    for (player, gravity_score) in offball:
                        if player not in playerDict[gameID]:
                            playerDict[gameID][player] = ([], []) 
                        playerDict[gameID][player][0].append(gravity_score)
                    for (player, gravity_score) in onball:
                        if player not in playerDict[gameID]:
                            playerDict[gameID][player] = ([], []) 
                        playerDict[gameID][player][1].append(gravity_score)"""
                    gravityDict[gameID].append(possDict)
                    possession = False
                    continue
                
                
                if row.event_id == 7:
                    if time not in possDict:
                        possDict[time] = {}
                    pairs = BarycentricPairing.newPairing(duke_location, opponent_location)
                    for d, o in pairs:
                        possDict[time][duke_players[o]] = [opponent_location[d], duke_location[o], ball, hoop]
                    possDict[time][ball_handler].append(0)
                    one, two, three, four, five, ball_gravity = TriangleInterpolation.find_average(possDict)
                    final_dict = process([one, two, three, four, five, ball_gravity])
                    playerDict[gameID].append(final_dict)
                    """if time not in possDict:
                        possDict[time] = {}
                    offball, onball = TriangleInterpolation.find_average(possDict)
                    pairs = BarycentricPairing.newPairing(duke_location, opponent_location)
                    for d, o in pairs:
                        possDict[time][duke_players[o]] = [opponent_location[d], duke_location[o], ball, hoop]
                    for (player, gravity_score) in offball:
                        
                        if player not in playerDict[gameID]:
                            playerDict[gameID][player] = ([], []) 
                        playerDict[gameID][player][0].append(gravity_score)
                    for (player, gravity_score) in onball:
                        if player not in playerDict[gameID]:
                            playerDict[gameID][player] = ([], []) 
                        playerDict[gameID][player][1].append(gravity_score)"""
                    gravityDict[gameID].append(possDict)
                    possession = False
                    continue
                
                
                if row.event_id == 15:
                    """if time not in possDict:
                        possDict[time] = {}
                    offball, onball = TriangleInterpolation.find_average(possDict)
                    pairs = BarycentricPairing.newPairing(duke_location, opponent_location)
                    for d, o in pairs:
                        possDict[time][duke_players[o]] = [opponent_location[d], duke_location[o], ball, hoop]
                    for (player, gravity_score) in offball:
                        if player not in playerDict[gameID]:
                            playerDict[gameID][player] = ([], []) 
                        playerDict[gameID][player][0].append(gravity_score)
                    for (player, gravity_score) in onball:
                        if player not in playerDict[gameID]:
                            playerDict[gameID][player] = ([], []) 
                        playerDict[gameID][player][1].append(gravity_score)"""
                    gravityDict[gameID].append(possDict)
                    possession = False
                    continue
                
                
                if row.event_id == 28 or row.event_id == 11:
                    continue
                if time not in possDict:
                    possDict[time] = {}
                ball = Coordinate(row.ball_x, row.ball_y)
                pairs = BarycentricPairing.newPairing(duke_location, opponent_location)
                for d, o in pairs:
                    possDict[time][duke_players[o]] = [opponent_location[d], duke_location[o], ball, hoop]
                possDict[time][ball_handler].append(0)
#                        possDict[time][duke_players[i]] = (opponents[i][0], opponents[i][1], ball, hoop)
    #print(gravityDict.items())
    #count = 0
    cleanDict = {}
    for game, possList in gravityDict.items():
        if game not in cleanDict:
            cleanDict[game] = []
        new_possList = [k for k in possList if 2.5 < max(k.keys()) - min(k.keys()) < 28]
        for d in new_possList:
            clean_d = {i: d[i] for i in d.keys() if len(d[i].keys()) == 5}
            cleanDict[game].append(clean_d)
    return cleanDict, shots#playerDict
    
def analyze_individual():
    cleanDict, playerDict = get_vectors()
    ret = []
    for game, d_list in playerDict.items():
        tempDict = {}
        for d in d_list:
            for k, v in d.items():
                if k not in tempDict:
                    tempDict[k] = [[] for i in range(10)]
                for j in range(10):
                    if len(v[j]) != 0:
                        tempDict[k][j].extend(v[j])

        for k, v in tempDict.items():
            avg_list = [(sum(v[i])/len(v[i]), len(v[i])) if len(v[i]) > 0 else 0 for i in range(10)]
            final_lst = [game, k]
            final_lst.extend(avg_list)
            ret.append(final_lst)
    """for game, players in playerDict.items():
        for player, gravity_scores in players.items():
            offball = sum(gravity_scores[0])/len(gravity_scores[0])
            onball = "N/A"
            if len(gravity_scores[1]) > 0:
                onball = sum(gravity_scores[1])/len(gravity_scores[1])
            ret.append([game, player, offball, onball])"""
    df = pd.DataFrame(ret, columns = ["game", "player", "offball_zone_one", "offball_zone_two", "offball_zone_three", "offball_zone_four", "offball_zone_five", "ball_one", "ball_two", "ball_three", "ball_four", "ball_five"])
    print(df)
    df.to_csv("distance_gravity_calculations_newest_bounds.csv")
    
if __name__ == '__main__':
    analyze_individual()
    """d, p = get_vectors()
    for lst in d[20150313]:
        print(max(lst.keys()), min(lst.keys()))"""

