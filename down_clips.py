import os
import json
import string
import random

import subprocess
import multiprocessing

with open('injury_data.json', 'r') as f:
        data = json.load(f)
        

def local_clip(filename, start_time, duration, output_filename):
    end_time = start_time + duration
    command = ['ffmpeg',
               '-i', '"%s"' % filename,
               '-ss', str(start_time),
               '-t', str(end_time - start_time),
               '-c:v', 'copy', '-an',
               '-threads', '1',
               '-loglevel', 'panic',
               '/ssd/ajpiergi/baseball/pitches/"%s"' % output_filename]
    command = ' '.join(command)

    try:
        output = subprocess.check_output(command, shell=True,
                                         stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as err:
        print(err.output)
        return err.output


def down_vid(yt, output):
    url_base='https://www.youtube.com/watch?v='
    command = ['/home/ajpiergi/anaconda2/bin/youtube-dl',#'youtube-dl',
               #               '--quiet', '--no-warnings',
               '-f', 'bestvideo[ext=mp4]',
               '-o', '"%s"' % output,
               '"%s"' % (url_base + yt)]
    command = ' '.join(command)
    print(command)
    #os.system(command)
                                                                    
def wrapper(pitch):
    local_clip('/ssd/ajpiergi/baseball/videos/'+file_.split('.')[0]+'.mkv', pitch['start']/fps, pitch['duration']/fps, pitch['clip_name']+'.mp4')
    return 0
    
yt_ids = []
for p in data['pitch_data']:
    yt_ids.append(p['yt'])

yt_ids = set(yt_ids)
print(yt_ids, len(yt_ids))


vid = '/ssd/ptin/vid'
for yt in yt_ids:
    # download yt video if doesnt exist
    if not os.path.exists(os.path.join(vid, yt+'.mp4')):
        down_vid(yt, os.path.join(vid, yt+'.mp4'))

exit()
cmds = ['/home/ajpiergi/anaconda2/bin/youtube-dl -f bestvideo[ext=mp4] -o "/ssd/ptin/vid/jD86dDAbMMg.mp4" "https://www.youtube.com/watch?v=jD86dDAbMMg"',
        '/home/ajpiergi/anaconda2/bin/youtube-dl -f bestvideo[ext=mp4] -o "/ssd/ptin/vid/tQkeOv-sJ3w.mp4" "https://www.youtube.com/watch?v=tQkeOv-sJ3w"',
        '/home/ajpiergi/anaconda2/bin/youtube-dl -f bestvideo[ext=mp4] -o "/ssd/ptin/vid/jVsLXDA6BLs.mp4" "https://www.youtube.com/watch?v=jVsLXDA6BLs"',
        '/home/ajpiergi/anaconda2/bin/youtube-dl -f bestvideo[ext=mp4] -o "/ssd/ptin/vid/zMa93aktzEY.mp4" "https://www.youtube.com/watch?v=zMa93aktzEY"',
        '/home/ajpiergi/anaconda2/bin/youtube-dl -f bestvideo[ext=mp4] -o "/ssd/ptin/vid/WQeMos2_mko.mp4" "https://www.youtube.com/watch?v=WQeMos2_mko"',
        '/home/ajpiergi/anaconda2/bin/youtube-dl -f bestvideo[ext=mp4] -o "/ssd/ptin/vid/R3dXKh5X9Oo.mp4" "https://www.youtube.com/watch?v=R3dXKh5X9Oo"',
        '/home/ajpiergi/anaconda2/bin/youtube-dl -f bestvideo[ext=mp4] -o "/ssd/ptin/vid/cylP1wYXag4.mp4" "https://www.youtube.com/watch?v=cylP1wYXag4"',
        '/home/ajpiergi/anaconda2/bin/youtube-dl -f bestvideo[ext=mp4] -o "/ssd/ptin/vid/gNvX_CMmOd4.mp4" "https://www.youtube.com/watch?v=gNvX_CMmOd4"',
        '/home/ajpiergi/anaconda2/bin/youtube-dl -f bestvideo[ext=mp4] -o "/ssd/ptin/vid/F7PWPw1fYiE.mp4" "https://www.youtube.com/watch?v=F7PWPw1fYiE"',
        '/home/ajpiergi/anaconda2/bin/youtube-dl -f bestvideo[ext=mp4] -o "/ssd/ptin/vid/_ZKFVvzIHGA.mp4" "https://www.youtube.com/watch?v=_ZKFVvzIHGA"',
        '/home/ajpiergi/anaconda2/bin/youtube-dl -f bestvideo[ext=mp4] -o "/ssd/ptin/vid/I0jNhfAZs-U.mp4" "https://www.youtube.com/watch?v=I0jNhfAZs-U"',
        '/home/ajpiergi/anaconda2/bin/youtube-dl -f bestvideo[ext=mp4] -o "/ssd/ptin/vid/GFDUgq0ae80.mp4" "https://www.youtube.com/watch?v=GFDUgq0ae80"',
        '/home/ajpiergi/anaconda2/bin/youtube-dl -f bestvideo[ext=mp4] -o "/ssd/ptin/vid/Os7i4IOAUBI.mp4" "https://www.youtube.com/watch?v=Os7i4IOAUBI"',
        '/home/ajpiergi/anaconda2/bin/youtube-dl -f bestvideo[ext=mp4] -o "/ssd/ptin/vid/AWmTrJItceA.mp4" "https://www.youtube.com/watch?v=AWmTrJItceA"',
        '/home/ajpiergi/anaconda2/bin/youtube-dl -f bestvideo[ext=mp4] -o "/ssd/ptin/vid/eA-Jooav_UA.mp4" "https://www.youtube.com/watch?v=eA-Jooav_UA"',
        '/home/ajpiergi/anaconda2/bin/youtube-dl -f bestvideo[ext=mp4] -o "/ssd/ptin/vid/gnEW5UZPYTY.mp4" "https://www.youtube.com/watch?v=gnEW5UZPYTY"',
        '/home/ajpiergi/anaconda2/bin/youtube-dl -f bestvideo[ext=mp4] -o "/ssd/ptin/vid/H-Iz_SzdXls.mp4" "https://www.youtube.com/watch?v=H-Iz_SzdXls"',
        '/home/ajpiergi/anaconda2/bin/youtube-dl -f bestvideo[ext=mp4] -o "/ssd/ptin/vid/gE-C6jtKV4Y.mp4" "https://www.youtube.com/watch?v=gE-C6jtKV4Y"',
        '/home/ajpiergi/anaconda2/bin/youtube-dl -f bestvideo[ext=mp4] -o "/ssd/ptin/vid/i9C83IubIw8.mp4" "https://www.youtube.com/watch?v=i9C83IubIw8"',
        '/home/ajpiergi/anaconda2/bin/youtube-dl -f bestvideo[ext=mp4] -o "/ssd/ptin/vid/vx4ooBUXdlA.mp4" "https://www.youtube.com/watch?v=vx4ooBUXdlA"',
        '/home/ajpiergi/anaconda2/bin/youtube-dl -f bestvideo[ext=mp4] -o "/ssd/ptin/vid/OrTuLYhmuQQ.mp4" "https://www.youtube.com/watch?v=OrTuLYhmuQQ"',
        '/home/ajpiergi/anaconda2/bin/youtube-dl -f bestvideo[ext=mp4] -o "/ssd/ptin/vid/a7paOawCa-I.mp4" "https://www.youtube.com/watch?v=a7paOawCa-I"',
        '/home/ajpiergi/anaconda2/bin/youtube-dl -f bestvideo[ext=mp4] -o "/ssd/ptin/vid/dNczsKnLHWg.mp4" "https://www.youtube.com/watch?v=dNczsKnLHWg"',
        '/home/ajpiergi/anaconda2/bin/youtube-dl -f bestvideo[ext=mp4] -o "/ssd/ptin/vid/h2HcdcpxYyE.mp4" "https://www.youtube.com/watch?v=h2HcdcpxYyE"',
        '/home/ajpiergi/anaconda2/bin/youtube-dl -f bestvideo[ext=mp4] -o "/ssd/ptin/vid/yVsspUDDHC0.mp4" "https://www.youtube.com/watch?v=yVsspUDDHC0"',
        '/home/ajpiergi/anaconda2/bin/youtube-dl -f bestvideo[ext=mp4] -o "/ssd/ptin/vid/AdAgo1spHho.mp4" "https://www.youtube.com/watch?v=AdAgo1spHho"',
        '/home/ajpiergi/anaconda2/bin/youtube-dl -f bestvideo[ext=mp4] -o "/ssd/ptin/vid/irLSdOmwAyg.mp4" "https://www.youtube.com/watch?v=irLSdOmwAyg"',
        '/home/ajpiergi/anaconda2/bin/youtube-dl -f bestvideo[ext=mp4] -o "/ssd/ptin/vid/x6vSIx9aF4I.mp4" "https://www.youtube.com/watch?v=x6vSIx9aF4I"']

def run(cmd):
   os.system(cmd)
pool = multiprocessing.Pool(processes=8)
pool.map(run, cmds)
   
exit()
for i in [0]:
    pool = multiprocessing.Pool(processes=8)
    pool.map(wrapper, data['game_data'])
    
