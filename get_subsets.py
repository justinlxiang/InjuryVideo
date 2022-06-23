import json
import sys

with open('new_inj_data.json', 'r') as f:
    data = json.load(f)


def get_by_key_val(key,val):
    pt = []
    for p in data['pitch_data']:
        if str(p[key]) == str(val):
            pt.append(p)
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
        i['inj'] = (1 if int(p['to_inj']) < k and int(p['to_inj']) > -10 else 0)
        n.append(i)
    return n
if __name__ == '__main__':
    print(last_k_as_injured(get_by_pitcher(sys.argv[1]+' '+sys.argv[2])))
#print get_by_arm(sys.argv[1])
#print get_by_inj_loc(sys.argv[1])
#print get_by_key_val(sys.argv[1],sys.argv[2])
