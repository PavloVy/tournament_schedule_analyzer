import numpy as np
import matplotlib.pyplot as plt

input_string = '''Gigolo	11:3	Nova QU
SkyForce	8:7	Gigolo Black
MadCaps W	11:1	Lynx W
Gigolo	10:5	Lynx
MadCaps	11:3	Gigolo Black
Dyki Gamble	11:1	SkyForce W
Вовчиці	8:7	Lynx W
Nova QU	6:8	Lynx
MadCaps	10:6	SkyForce
Dyki Gamble	10:4	Вовчиці
Nova	11:4	NXT2
SkyForce Legacy	1:11	NXT
Dyki Gamble	11:6	MadCaps W
Вовчиці	8:2	SkyForce W
Nova	11:5	Wolves
FirePlay	4:8	NXT
NXT2	11:5	Wolves
FirePlay	11:3	SkyForce Legacy
Nova	11:3	SkyForce
MadCaps	4:8	NXT2
Gigolo	11:6	FirePlay
NXT	10:4	Lynx
Wolves	10:8	Gigolo Black
Nova QU	10:6	SkyForce Legacy
SkyForce W	2:11	Lynx W
Вовчиці	4:8	MadCaps W
SkyForce Legacy	6:11	Gigolo Black
Wolves	10:11	Nova QU
Dyki Gamble	8:5	Lynx W
SkyForce W	0:11	MadCaps W
Wolves	11:4	SkyForce Legacy
Nova QU	6:9	Gigolo Black
SkyForce	9:6	FirePlay
MadCaps	10:9	Lynx
Nova	10:5	NXT2
Gigolo	8:7	NXT
FirePlay	6:9	Lynx
SkyForce	8:11	MadCaps
Вовчиці	6:7	Lynx W
NXT	9:7	NXT2
Dyki Gamble	15:9	MadCaps W
Nova	15:10	Gigolo'''

optimization_step = 0.01

class Team:
    def __init__(self, name):
        self.name = name
        self.nodes = []
        self.level = 0

    def add_connection(self,connection):
        self.nodes.append(connection)

    def __repr__(self):
        return self.name +': ' + str(self.level)

class Connection:
    def __init__(self,teamA:Team, teamB:Team, scoredA:int, scoredB:int):
        self.A = teamA
        self.A.add_connection(self)
        self.B = teamB
        self.B.add_connection(self)
        self.scoreA = scoredA
        self.scoreB = scoredB

    def give_other(self,this):
        if this == self.A:
            return self.B
        elif this == self.B:
            return self.A
        else:
            print('Wrong query. Will give you None')
            return None

    def step_equalize(self):
        target_diff = self.scoreA - self.scoreB
        observed_diff = self.A.level - self.B.level
        mismatch = target_diff - observed_diff
        addon = mismatch*optimization_step/2
        self.A.level+=addon
        self.B.level-=addon

    def __repr__(self):
        return self.A.name + ' ' + str(self.scoreA) + ' ' + self.B.name + ' '+ str(self.scoreB)




def main():
    strings = input_string.split('\n')

    games = []
    for s in strings:
        parts = s.split('	')
        counts = [int(n) for n in parts[1].split(':')]
        games.append([parts[0],parts[2],counts])

    teams_dict = {}


    for game in games:
        if game[0] not in teams_dict.keys():
            teams_dict[game[0]] = Team(game[0])
        if game[1] not in teams_dict.keys():
            teams_dict[game[1]] = Team(game[1])

    # sk = sorted([k in teams_dict.keys()])
    # for s in sk:
    #     print(s)
    connections = []
    for game in games:
        connections.append(Connection(teams_dict[game[0]],teams_dict[game[1]],game[2][0],game[2][1]))

    women_list = ['Dyki Gamble']
    for node in teams_dict['Dyki Gamble'].nodes:
        other = node.give_other( teams_dict['Dyki Gamble'])
        if not(other.name in women_list):
            women_list.append(other.name)
    print(women_list)

    for i in range(1500):
        for con in connections:
            con.step_equalize()

    teams_list = []
    for k in teams_dict.keys():
        teams_list.append(teams_dict[k])

    plotlist = []
    tls = sorted(teams_list, key = lambda s: s.level, reverse= True)
    for t in tls:
        if t.name not in women_list:
            print(t)
            plotlist.append(t.level)

    plt.plot(plotlist)
    plt.show()

    print('-'*20)

    for t in tls:
        if t.name in women_list:
            print(t)


if __name__ == '__main__':
    main()