import json
import os, sys

with open('new_inj_data.json', 'r') as f:
    data = json.load(f)


# def get_by_key_val(key,val):
def get_by_key_val(key):
    pitchers = ["Aaron Nola", "Clayton Kershaw", "Corey Kluber", "A.J. Griffin", "Adalberto Mejia", "Adam Liberatore", "Adam Wainwright", "Alex Meyer", "Alex Wood", "Andrew Bailey", "Aroldis Chapman", "Austin Brice", "Boone Logan", "Brandon Finnegan", "Brent Suter", "Brett Anderson"]
    pt = []
    for pitcher in pitchers:
        pid = data['players'][pitcher]
        count = 0
        for p in data['pitch_data']:
            # if str(p[key])== str(val):
            if pid==p['pitcher'] and os.path.isfile("/home/ec2-user/InjuryVideo/RGB/" + str(p[key].upper() +".npy")): 
                pt.append(p)
                count += 1
        print(pitcher + " " + str(count))
    return pt
                                            
def get_by_pitcher(name):
    pid = data['players'][name]
    return get_by_key_val('pitcher', pid)
    #for p in data['pitch_data']:
    #    if p['pitcher'] == pid:
    #        pt.append(p)
    #return pt

def get_by_arm(arm):
    return get_by_key_val('arm',arm)
    #pt = []
    #for p in data['pitch_data']:
    #    if p['arm'] == arm:
    #        pt.append(p)
    #return pt

def get_by_inj_loc(loc):
    return get_by_key_val('inj_loc', loc)
    #pt = []
    #for p in data['pitch_data']:
    #    if p['inj_loc'] == loc:
    #        pt.append(p)
    #return pt

def get_by_inj_id(i):
    return get_by_key_val('inj_id', i)


def last_k_as_injured(d, k=10):
    n = []
    for p in d:
        i = {}
        i['clip_name'] = p['clip_name']
        i['inj'] = (1 if int(p['to_inj']) > 0 and int(p['to_inj']) > -10 else 0)
        n.append(i)
    return n
if __name__ == '__main__':
    print(last_k_as_injured(get_by_pitcher(sys.argv[1]+' '+sys.argv[2])))
#print get_by_arm(sys.argv[1])
#print get_by_inj_loc(sys.argv[1])
#print get_by_key_val(sys.argv[1],sys.argv[2])
