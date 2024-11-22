from abc import ABC, abstractmethod
import numpy as np
from typing import List
import cv2 , random
import matplotlib.pyplot as plt
from sympy.physics.units import temperature

line_step = 25
temperature = 0

class Team:
    def __init__(self, name: str, power: int):
        self.name = name
        self.power = power
        self.must_place  = None
        self.games = 0
        self.wins = 0
        self.diff = 0
        self.log = []
        self.compl_diff = 0
        self.teamlist = []

    def set_must(self,place):
        self.must_place = place

    def set_game(self,win,diff,who):
        self.games+=1
        self.wins+=win
        self.diff+=diff
        self.log.append([who,diff])
        self.teamlist.append([who.name])

    def recalculate(self):
        self.compl_diff = 0
        for element in self.log:
            who = element[0]
            dif = element[1]
            who_hardness = np.exp(who.diff)
            own_performance = np.exp(dif)

            self.compl_diff += who_hardness*own_performance

    def reset(self):
        self.games = 0
        self.wins = 0
        self.diff = 0
        self.log = []

    def __repr__(self):
        return f"{self.name} (Power: {self.power}, Place: {self.must_place}), (Games: {self.games}, Wins: {self.wins}, Dif: {self.compl_diff})"

class Connection:
    def __init__(self, name, to_what = None, coord = None):
        self.name = name
        self.connected_to = to_what
        self.message = None
        self.coord = coord


    def connect(self, to_what):
        if self.connected_to is None:
            self.connected_to = to_what
        else:
            print('WARNING! DOUBLE CONNECTION.')

    def check_connection(self):
        return not(self.connected_to is None)

    def propagate(self,message):
        self.connected_to.receive(message)

    def receive(self,message):
        self.message = message


class Connection_manager:
    def __init__(self):
        self.objects = []
        self.all_inputs = []
        self.all_outputs = []

    def register_block(self, obj):
        self.objects.append(obj)
        if obj.inputs is not None:
            self.all_inputs = self.all_inputs + obj.inputs
        if obj.outputs is not None:
            self.all_outputs = self.all_outputs + obj.outputs

    def check_everything(self):
        for elements_list in [self.all_inputs, self.all_outputs]:
            for i in elements_list:
                if not (i.check_connection()):
                    print(i.name + ' not connected')

                # if not(i.message is None):
                #     print(i.message)

    def count_games(self):
        games = 0
        for o in self.objects:
            games+=o.num_games
        print('Total games',games)

    def fill_teams(self,teams):
        self.objects[0].fill_teams(teams)

    def run_full_sim(self, randomize = False):
        if randomize:
            random.shuffle(self.objects[0].outputs)

        for team in self.objects[0].teams:
            team.reset()

        for o in self.objects:
            o.make_play()

        return self.objects[-1].report_placement()

    def represent_sel(self):
        image = np.zeros((1024,1920,3),dtype=np.uint8)
        for o in self.objects:
            if o.outputs is not None:
                for out in o.outputs:
                    cv2.line(image,out.coord,out.connected_to.coord,[233,200,120],3)
                sz = len(o.outputs)
            else:
                sz = len(o.inputs)
            cv2.rectangle(image,o.coord,[o.coord[0]+10,o.coord[1]+line_step*sz],o.color,3)

        cv2.imshow('w',image)
        cv2.waitKey(0)



class Source:
    def __init__(self,size, coord):
        self.outputs = []
        self.inputs = None
        self.teams = None
        self.num_games = 0
        for i in range(size):
            out_obj = Connection('Source '+ str(i), coord = [coord[0],coord[1]+i*line_step] )
            self.outputs.append(out_obj)

        self.coord = coord
        self.color = [255,100,100]

    def fill_teams(self,teams):
        self.teams = teams

    def make_play(self):
        for team,connection in zip(self.teams,self.outputs):
            connection.propagate(team)

    def pack(self,cm):
        cm.register_block(self)
        return self


class Final_standing:
    def __init__(self, size , coord):
        self.inputs = []
        self.outputs = None
        self.num_games = 0
        for i in range(size):
            input_obj = Connection('Standing '+ str(i), coord = [coord[0],coord[1]+i*line_step]  )
            self.inputs.append(input_obj)
        self.coord = coord
        self.color = [100, 255,  100]

    def report_placement(self):
        placement = []
        for inp in self.inputs:
            # print(inp.message)
            placement.append(inp.message)
        return placement

    def make_play(self):
        #self.report_placement()
        pass

    def pack(self,cm):
        cm.register_block(self)
        return self

class Game_block(ABC):
    def __init__(self,size,coord):
        self.inputs = []
        self.outputs = []
        self.num_games = None
        self.standings = None
        for i in range(size):
            input_obj = Connection('Game in ' + str(i), coord = [coord[0]-35,coord[1]+i*line_step] )
            self.inputs.append(input_obj)
            out_obj = Connection('Game out ' + str(i), coord = [coord[0]+35,coord[1]+i*line_step])
            self.outputs.append(out_obj)
        self.color = None
        self.coord = coord

    def pack(self,cm):
        cm.register_block(self)
        return self

    @abstractmethod
    def sort_teams(self):
        pass

    @abstractmethod
    def make_play(self):

        self.sort_teams()
        for team,out in zip(self.standings, self.outputs):
            out.propagate(team)


class Rr(Game_block):
    def __init__(self,size, coord):
        super().__init__(size, coord)
        self.num_games = size*(size-1)/2
        self.color = [255, 0, 100]

    def sort_teams(self):

        gathered = []
        for inp in self.inputs:
            gathered.append(inp.message)
        wins, difs = np.zeros((len(self.inputs,))),np.zeros((len(self.inputs,)))
        for i in range(len(gathered) - 1):
            teamA = gathered[i]
            for k in range(i + 1, len(gathered)):
                teamB = gathered[k]
                win, diff = play_game(teamA, teamB)
                if win==0:
                    wins[i]+=1
                else:
                    wins[k]+=1
                difs[i] += diff
                difs[k] -= diff

        sorting_idx = np.flipud(np.argsort(wins))
        self.standings = []
        for s in sorting_idx:
            self.standings.append(gathered[s])


    def make_play(self):
        super().make_play()
        pass

class One_game(Game_block):
    def __init__(self,coord):
        size = 2
        super().__init__(size,coord)
        self.num_games = 1
        self.color = [255, 0, 0]

    def sort_teams(self):
        teamA = self.inputs[0].message
        teamB = self.inputs[1].message
        win, diff = play_game(teamA, teamB)
        self.standings = []
        if win == 0:
            self.standings.append(teamA)
            self.standings.append(teamB)
        else:
            self.standings.append(teamB)
            self.standings.append(teamA)

    def make_play(self):
        super().make_play()
        pass

class Seeding_block(Game_block):
    def __init__(self,size,temp,coord):
        super().__init__(size, coord)
        self.num_games = 0
        self.color = [255, 255,255]
        self.temperature = temp

    def sort_teams(self):
        gathered = []
        for inp in self.inputs:
            gathered.append(inp.message)

        perceived_power = []
        for team in gathered:
            perceived_power.append(team.power)+ np.random.randn(1)[0] * self.temperature



        self.standings = seed

    def make_play(self):
        super().make_play()
        pass

class Olympic_block(Game_block):
    def __init__(self,size, coord):
        super().__init__(size, coord)
        self.num_games = size*np.log2(size)/2

        self.color = [ 0, 255, 100]

    def sort_teams(self):
        def olympic_round(teams):
            winners = []
            loosers = []
            for i in range(len(teams)//2):
                t1 = teams[i*2]
                t2 = teams[i*2+1]
                win,diff= play_game(t1,t2)
                if win==0:
                    winners.append(t1)
                    loosers.append(t2)
                else:
                    winners.append(t2)
                    loosers.append(t1)
            return winners,loosers

        def recursive_olypmic(tt):
            w,l = olympic_round(tt)
            if len(w)==1:
                return w,l
            else:
                ww,wl = recursive_olypmic(w)
                lw,ll = recursive_olypmic(l)
                return ww+wl,lw+ll

        gathered = []
        for inp in self.inputs:
            gathered.append(inp.message)

        w,l = recursive_olypmic(gathered)
        self.standings = w+l

    def make_play(self):
        super().make_play()
        pass

class Swiss_block(Game_block):
    def __init__(self,size, coord):
        super().__init__(size, coord)
        self.num_games = size/2

        self.color = [ 0, 0, 200]

    def sort_teams(self):
        messages = [inp.message for inp in self.inputs]
        for message in messages:
            message.recalculate()

        taken_messages = messages.copy()
        while len(taken_messages)>0:
            current_mes = taken_messages.pop(0)
            found = False
            for i in range(len(taken_messages)):
                if not(current_mes.name in taken_messages[i].teamlist):
                    found = True
                    second_mes = taken_messages.pop(i)
                    break
            if not(found):
                second_mes = taken_messages.pop(0)
            play_game(current_mes,second_mes)

        # for i in range(len(self.inputs)//2):
        #     t1 = messages[i*2]
        #     t2 = messages[i*2+1]
        #     win,diff= play_game(t1,t2)

        # for m in messages:
        #     print(m)
        # print('===================')


        sorted_teams = sorted(messages,key= lambda x: x.wins+x.diff/100000000000,reverse=True)

        self.standings = sorted_teams


    def make_play(self):
        super().make_play()
        pass

def create_connection(what: Connection, to_what: Connection):
    what.connect(to_what = to_what)
    to_what.connect(to_what=what)

def link_lists(l1,l2):
    if type(l1) == list:
        if len(l1)!=len(l2):
            print('WARNING, WRONG SIZES')
        else:
            for e1,e2 in zip(l1,l2):
                create_connection(e1,e2)
    else:
        create_connection(l1, l2)

def play_game(A:Team,B:Team):
    local_B_power = B.power + np.random.randn(1)[0] * temperature
    dif = int((A.power - local_B_power) / 10)

    if A.power>local_B_power:
        win = 0
    else:
        win = 1

    if dif==0:
        dif=-win*2+1

    A.set_game(1-win,dif,B)
    B.set_game(win,-dif,A)

    return win,dif


def swiss(nteams,stages):
    cm = Connection_manager()


    source = Source(nteams, [100, 200]).pack(cm)
    prev_ous = source.outputs
    for i in range(stages):
        sb = Swiss_block(nteams,[(i+1)*100,200]).pack(cm)
        link_lists(prev_ous,sb.inputs)
        prev_ous = sb.outputs
    sink = Final_standing(nteams,[(i+3)*100,200]).pack(cm)
    link_lists(prev_ous,sink.inputs)

    cm.check_everything()
    cm.count_games()

    return cm

def swiss_hybrid(nteams,stages):
    cm = Connection_manager()


    source = Source(nteams, [100, 200]).pack(cm)
    prev_ous = source.outputs
    for i in range(stages):
        sb = Swiss_block(nteams,[(i+1)*100,200]).pack(cm)
        link_lists(prev_ous,sb.inputs)
        prev_ous = sb.outputs

    ols = []
    for i in range(3):
        ol = Olympic_block(4,[700,(2+i)*150]).pack(cm)
        link_lists(prev_ous[i*4:(i+1)*4],ol.inputs)
        ols.append(ol)

    sink = Final_standing(nteams,[900,200]).pack(cm)
    for i in range(3):
        ol = ols[i]
        link_lists(ol.outputs, sink.inputs[i*4:(i+1)*4])


    cm.check_everything()
    cm.count_games()

    return cm

def swiss_olympic(nteams,stages):
    cm = Connection_manager()


    source = Source(nteams, [100, 200]).pack(cm)
    prev_ous = source.outputs
    for i in range(stages):
        sb = Swiss_block(nteams,[(i+1)*100,200]).pack(cm)
        link_lists(prev_ous,sb.inputs)
        prev_ous = sb.outputs


    olbig = Olympic_block(8,[700,150]).pack(cm)
    olsmall = Olympic_block(4,[700,450]).pack(cm)
    link_lists(prev_ous[:8],olbig.inputs)
    link_lists(prev_ous[8:], olsmall.inputs)


    sink = Final_standing(nteams,[900,200]).pack(cm)

    link_lists(olbig.outputs, sink.inputs[:8])
    link_lists(olsmall.outputs, sink.inputs[8:])


    cm.check_everything()
    cm.count_games()

    return cm

def konukh_net():
    cm = Connection_manager()
    source = Source(12, [100, 200]).pack(cm)
    round_robinA = Rr(3, [300, 100]).pack(cm)
    round_robinB = Rr(3, [300, 300]).pack(cm)
    round_robinC = Rr(3, [300, 500]).pack(cm)
    round_robinD = Rr(3, [300, 700]).pack(cm)

    crossA1D2 = One_game(coord = [500,100]).pack(cm)
    crossD1A2 = One_game(coord=[500, 200]).pack(cm)

    crossC1B2 = One_game(coord=[500, 300]).pack(cm)
    crossB1C2 = One_game(coord=[500, 400]).pack(cm)

    round_robin_lower = Rr(4,[600, 750]).pack(cm)

    s1 = One_game(coord=[700, 400]).pack(cm)
    s2 = One_game(coord=[700, 300]).pack(cm)
    gam5 = One_game(coord=[700, 200]).pack(cm)
    gam6 = One_game(coord=[700, 100]).pack(cm)


    final78 = One_game(coord=[900, 400]).pack(cm)
    final56 = One_game(coord=[900, 300]).pack(cm)
    final34 = One_game(coord=[900, 200]).pack(cm)
    final12 = One_game(coord=[900, 100]).pack(cm)

    sink = Final_standing(12,coord = [1200,200]).pack(cm)


    # blocks = [source,round_robinA,round_robinB,round_robinC,round_robinD, crossA1D2,crossD1A2,crossC1B2,crossB1C2]
    # blocks+=[s1,s2,gam5,gam6,final78,final56,final34,final12]
    # blocks+=[round_robin_lower, sink]
    # for block in blocks:
    #     cm.register_block(block)

    for i,robin in enumerate([round_robinA,round_robinB,round_robinC,round_robinD]):
        link_lists(source.outputs[i*3:(i+1)*3],robin.inputs)

    last_places = [x.outputs[2] for x in [round_robinA,round_robinB,round_robinC,round_robinD]]
    link_lists(last_places, round_robin_lower.inputs)
    link_lists(round_robin_lower.outputs,sink.inputs[8:])

    link_lists(round_robinA.outputs[0],crossA1D2.inputs[0])
    link_lists(round_robinD.outputs[1], crossA1D2.inputs[1])

    link_lists(round_robinA.outputs[1], crossD1A2.inputs[0])
    link_lists(round_robinD.outputs[0], crossD1A2.inputs[1])

    link_lists(round_robinC.outputs[0], crossC1B2.inputs[0])
    link_lists(round_robinB.outputs[1], crossC1B2.inputs[1])

    link_lists(round_robinC.outputs[1], crossB1C2.inputs[0])
    link_lists(round_robinB.outputs[0], crossB1C2.inputs[1])

    link_lists(crossA1D2.outputs[1], s1.inputs[0])
    link_lists(crossC1B2.outputs[1], s1.inputs[1])

    link_lists(crossD1A2.outputs[1], s2.inputs[0])
    link_lists(crossB1C2.outputs[1], s2.inputs[1])

    link_lists(crossA1D2.outputs[0], gam5.inputs[0])
    link_lists(crossD1A2.outputs[0], gam5.inputs[1])

    link_lists(crossC1B2.outputs[0], gam6.inputs[0])
    link_lists(crossB1C2.outputs[0], gam6.inputs[1])

    link_lists(s1.outputs[1], final78.inputs[0])
    link_lists(s2.outputs[1], final78.inputs[1])

    link_lists(s1.outputs[0], final56.inputs[0])
    link_lists(s2.outputs[0], final56.inputs[1])

    link_lists(gam5.outputs[1], final34.inputs[0])
    link_lists(gam6.outputs[1], final34.inputs[1])

    link_lists(gam5.outputs[0], final12.inputs[0])
    link_lists(gam6.outputs[0], final12.inputs[1])

    list_of_places = final12.outputs+final34.outputs+final56.outputs+final78.outputs

    link_lists(list_of_places,sink.inputs[:8])

    cm.check_everything()
    cm.count_games()
    return cm

if __name__ == '__main__':

    # manager  = swiss(12,5)
    manager = swiss_hybrid(12,3)
    # manager = swiss_olympic(12,2)
    manager = konukh_net()

    manager.represent_sel()

    teams8 = [Team("Team A", 10),
             Team("Team B", 20),
             Team("Team C", 30),
             Team("Team d", 40),
             Team("Team e", 50),
             Team("Team f", 60),
             Team("Team g", 70),
             Team("Team h", 80)]

    teams12 = teams8 + [Team("Team i", 90),
             Team("Team j", 100),
             Team("Team k", 110),
             Team("Team l", 120)]

    teams = teams12

    t_sorted = sorted(teams, reverse=True, key = lambda x: x.power)
    for i,t in enumerate(t_sorted):
        t.set_must(i)

    manager.fill_teams(teams)
    num_sims = 1000
    errors = np.zeros((num_sims,len(teams),len(teams)),dtype=int)
    counter = 0
    err_counter = 0
    for s in range(num_sims):
        results = manager.run_full_sim(randomize=True)
        # for r in results:
        #     print(r)
        #     print([w[0].name for w in r.log])
        # input()
        for i,result in enumerate(results):
            error = abs(i-result.must_place)
            # if i==0 and error>0:
            #     for r in results:
            #         print(r)
            #     input()
            errors[s,i,error]+=1
            counter+=1
            err_counter+=error
    print('Mean err ', err_counter/counter)
    plt.imshow(np.mean(errors[:,:,1:],0))
    plt.figure()
    for i in range(len(teams)):
        if i>0:
            piece = errors[:,:,i]
            if np.sum(piece)>0:
                plt.plot(np.arange(errors.shape[1])+1,np.mean(piece,axis = 0),label=str(i)+' places error')
    plt.legend()
    plt.xlabel('Placement after the tournament')
    plt.ylabel('Fraction of placement error')
    plt.show()

