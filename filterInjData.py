import json
import os

with open('new_inj_data.json') as f:
    obj = json.load(f)

print(len(obj['pitch_data']))

idxs = []

for i in range(len(obj['pitch_data'])):
    print(i)
    if not os.path.exists(os.path.join('/Users/juxiang/Documents/InjuryVideo/RGB/', obj['pitch_data'][i]['clip_name'] + '.npy')):
        print(i, len(obj['pitch_data']))
        idxs.append(i)

for i in reversed(idxs):
    obj['pitch_data'].pop(i)

with open('filtered.json','w') as out:
    json.dump(obj,out)

print(len(obj['pitch_data']))