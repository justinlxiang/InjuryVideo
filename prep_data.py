import os
import sys
import json
import urllib2
import random
import string
import pandas

import xmltodict
import datetime

inj_data = pandas.read_csv('pitcher_injury_data_v1.csv')
#Index([u'Name', u'first', u'last', u'season', u'team', u'position', u'start',
#              u'end', u'days', u'location', u'type', u'side', u'Injury ID',
#              u'FG Teamid', u'FG playerid', u'mlbamid', u'Game Hurt', u'XML',
#              u'YouTube', u'First Pitch', u'Healthy XML', u'Healthy YT',
#              u'First Pitch.1'],
#            dtype='object')


def proc_game(url, first_pitch):
    FMT = '%Y-%m-%dT%H:%M:%S'
    duration = 10
    f = urllib2.urlopen(url+'players.xml')
    xml = f.read()
    f.close()
    data = xmltodict.parse(xml)

    player_ids = {}
    rev_pid = {}
    for team in data['game']['team']:
        for player in team['player']:
            player_ids[player['@first']+' '+player['@last']] = player['@id']
            rev_pid[player['@id']] = player['@first']+' '+player['@last']

        
    f = urllib2.urlopen(url+'inning/inning_all.xml')
    xml = f.read()
    f.close()
    pitch_data = xmltodict.parse(xml)
    
    game_data = []
    inning_i = 0
    inn = 'top'
    batter_i = 0
    pitch_i = 0
    last_pitch = 0
    
    st = pitch_data['game']['inning'][0]['top']['atbat'][0]['pitch']
    if type(st) == list:
        st = st[0]['@tfs_zulu'][:-1]
        gst = st
    else:
        st = st['@tfs_zulu']
        gst = st[:-1]#.split('T')[1]    
    start_time = first_pitch.split(':')
    start_time = int(start_time[0])*60*60 + int(start_time[1])*60 + int(start_time[2])

    while inning_i < len(pitch_data['game']['inning']):
        if inn not in pitch_data['game']['inning'][inning_i]:
            break
        if 'pitch' in pitch_data['game']['inning'][inning_i][inn]['atbat'][batter_i]:
            ptc = pitch_data['game']['inning'][inning_i][inn]['atbat'][batter_i]['pitch']
            bat = pitch_data['game']['inning'][inning_i][inn]['atbat'][batter_i]
        
            if type(ptc) == list:
                q = ptc[pitch_i]
                pitch_i += 1
                if pitch_i >= len(ptc):
                    pitch_i = 0
                    batter_i += 1
            else:
                q = ptc
                batter_i += 1
                pitch_i = 0
            
            st = q['@tfs_zulu']
            p2 = st[:-1]#.split('T')[1]
            td = datetime.datetime.strptime(p2, FMT) - datetime.datetime.strptime(gst, FMT)
            if len(game_data) == 0:
                print td.total_seconds(), start_time
            start_s = td.total_seconds() + start_time
            if last_pitch != 0:
                if start_s - last_pitch < 8:
                    print 'gmm'
            last_pitch = start_s
            p = {'start': start_s, 'duration': duration}
            #print p
            #p['url'] = urls[vid]
            
        
            q['@batter'] = bat['@batter']
            q['@pitcher'] = bat['@pitcher']
            q['@stand'] = bat['@stand']
            q['@p_throws'] = bat['@p_throws']
            q['@des_bat'] = bat['@des']
            q['@event_bat'] = bat['@event']
            q['@b'] = bat['@b']
            q['@s'] = bat['@s']
            q['@o'] = bat['@o']
            q['@inning'] = inning_i+1
            q['@inn'] = inn
            rnd_nm = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(12))
            q['@clip_name'] = rnd_nm
            for k in q.keys():
                p[k[1:]] = q[k]
            
            game_data.append(p)
        else:
            batter_i += 1
        if batter_i >= len(pitch_data['game']['inning'][inning_i][inn]['atbat']):
            batter_i = 0
            if inn == 'top':
                inn = 'bottom'
            else: 
                inn = 'top'
                inning_i += 1
                if inning_i >= len(pitch_data['game']['inning']):
                    print 'Done?'
                    break
                
    f = {'game_data':game_data, 'players':player_ids, 'rev_pid':rev_pid}
    return f


injury_data = {'pitch_data':[], 'players':{}, 'rev_pid':{}}
for injury_i in range(len(inj_data)):

    url = inj_data['XML'][injury_i]
    url2 = inj_data['Healthy XML'][injury_i]
    first_pitch = inj_data['First Pitch'][injury_i]
    first_pitch_healthy = inj_data['First Pitch.1'][injury_i]
    hurt_id = inj_data['mlbamid'][injury_i]
    arm = inj_data['position'][injury_i][0]
    days = inj_data['days'][injury_i]
    inj_loc = inj_data['location'][injury_i]
    inj_type = inj_data['type'][injury_i]
    side = inj_data['side'][injury_i]
    inj_id = inj_data['Injury ID'][injury_i]
    inj_yt = inj_data['YouTube'][injury_i]
    healthy_yt = inj_data['Healthy YT'][injury_i]
    

    inj_game = proc_game(url, first_pitch)
    injury_data['players'].update(inj_game['players'])
    injury_data['rev_pid'].update(inj_game['rev_pid'])

    new_pd = []
    pc = 0
    for pitch in inj_game['game_data']:
        if str(pitch['pitcher']) == str(hurt_id):
            pc += 1

    i = 0
    for pitch in inj_game['game_data']:
        if str(pitch['pitcher']) == str(hurt_id):
            pitch['to_inj'] = pc-i
            pitch['arm'] = arm
            pitch['days'] = days
            pitch['inj_loc'] = inj_loc
            pitch['inj_type'] = inj_type
            pitch['side'] = side
            pitch['inj_id'] = inj_id
            pitch['yt'] = inj_yt
            injury_data['pitch_data'].append(pitch)
            i += 1
            
    if url2 == '0':
        continue
    healthy_game = proc_game(url2, first_pitch_healthy)
    injury_data['players'].update(healthy_game['players'])
    injury_data['rev_pid'].update(healthy_game['rev_pid'])
        
    for pitch in healthy_game['game_data']:
        pitch['to_inj'] = -100
        pitch['arm'] = arm
        pitch['days'] = -1
        pitch['inj_loc'] = 'none'
        pitch['inj_type'] = 'none'
        pitch['side'] = 'none'
        pitch['inj_id'] = -1
        pitch['yt'] = healthy_yt
        injury_data['pitch_data'].append(pitch)

with open('injury_data_tmp.json','w') as out:
    json.dump(injury_data, out)
