import json

with open('filtered.json', 'r') as f:
    data = json.load(f)

names = ["Aaron Nola",
         "Clayton Kershaw",
         "Corey Kluber",
         "A.J. Griffin",
         "Adalberto Mejia",
         "Adam Liberatore",
         "Adam Wainwright",
         "Alex Meyer",
         "Alex Wood",
         "Andrew Bailey",
         "Aroldis Chapman",
         "Austin Brice",
         "Boone Logan",
         "Brandon Finnegan",
         "Brent Suter",
         "Brett Anderson"]

total = 0

for name in names:    
    tot = healthy = inj = 0
    for p in data['pitch_data']:
        if(data['players'][name] == p['pitcher']):
            tot += 1
            # if(p['to_inj'] < 20 and int(p['to_inj']) > -10):
            if(p['to_inj'] != -100):

                inj += 1
            else:
                healthy += 1
    print(name, tot, 'healthy:', healthy, 'inj:', inj)
    total += tot
print(total)
listnames = []

key_list = list(data['players'].keys())
val_list = list(data['players'].values())
arr = []
for p in data['pitch_data']:
    listnames.append(p['pitcher'])
    if(p['pitcher'] in val_list):
        position = val_list.index(p['pitcher'])
        arr.append(key_list[position])
print(len(set(arr)))
print(set(arr))

total = 0
toth = 0
toti = 0
for name in set(arr):    
    tot = healthy = inj = 0
    for p in data['pitch_data']:
        if(data['players'][name] == p['pitcher']):
            tot += 1
            if(p['to_inj'] < 20 and int(p['to_inj']) > -10):
                inj += 1
                toti += 1
            else:
                healthy += 1
                toth += 1
    print(name, tot, 'healthy:', healthy, 'inj:', inj)
    total += tot
print(total)
print(toth, toti)
print(len(set(arr)))