import json

with open('injury_data.json') as f:
    od = json.load(f)

with open('injury_data_tmp.json') as f:
    td = json.load(f)

npd = []
for x in td['pitch_data']:
    for y in od['pitch_data']:
        if sorted(x.keys()) == sorted(y.keys()):
            t = True
            for k in x.keys():
                if k == 'clip_name' or k == 'to_inj':
                    continue
                if x[k] != y[k]:
                    t = False
                    break
            if t:
                y['to_inj'] = x['to_inj']
                npd.append(y)
                break

od['pitch_data'] = npd
with open('new_inj_data.json','w') as out:
    json.dump(od,out)
