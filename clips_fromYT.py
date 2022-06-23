import os
import json
import string
import random

import subprocess
import multiprocessing

with open('new_inj_data.json', 'r') as f:
        data = json.load(f)

# crops = {'d':'600:460:300:190',
#          'vx4ooBUXdlA': '600:460:200:190',
#          'H-Iz_SzdXls': '600:460:140:190',
#          'F7PWPw1fYiE':'600:460:100:190',
#          'AdAgo1spHho': '600:460:180:180',
#          'dNczsKnLHWg':'600:460:230:190',
#          'AWmTrJItceA':'600:460:280:190',
#          'EuvlwwrqAtA':'600:460:240:190',
#          'zMa93aktzEY':'600:460:230:190',
#          'jD86dDAbMMg':'600:460:240:190',
#          'dNczsKnLHWg': '600:460:230:190',
#          'gnEW5UZPYTY': '600:460:250:190',
#          'XOdpZqbH58k': '600:460:310:200',
#          'tQkeOv-sJ3w': '600:460:260:200',
#          'I0jNhfAZs-U': '600:460:280:200',
#          'GFDUgq0ae80': '600:460:280:200',
#          'eA-Jooav_UA': '600:460:260:200',
#          '_ZKFVvzIHGA': '600:460:290:200',
#          'R3dXKh5X9Oo': '600:460:210:200',
#          'AdAgo1spHho': '600:460:260:200',
#          'WQeMos2_mko':'600:460:220:200',
#          'AWmTrJItceA': '600:460:210:200',
#          'Os7i4IOAUBI': '600:460:280:200',
#          'a7paOawCa-I': '600:460:310:190',
# }
        

# def local_clip(filename, start_time, duration, output_filename):
#     end_time = start_time + duration
#     vid = filename.split('/')[-1].split('.')[0]
#     crop = crops[vid] if vid in crops else crops['d']
#     command = ['ffmpeg',
#                '-i', '"%s"' % filename,
#                '-ss', str(start_time),
#                '-t', str(end_time - start_time),
#                '-c:v', 'copy', '-an',
#                '-filter:v "crop='+crop+'"',
#                '-threads', '1',
#                '-loglevel', 'panic',
#                '/ssd/ajpiergi/baseball/pitches/"%s"' % output_filename]
#     command = ' '.join(command)

#     try:
#         output = subprocess.check_output(command, shell=True,
#                                          stderr=subprocess.STDOUT)
#     except subprocess.CalledProcessError as err:
#         print(err.output)
#         return err.output


def down_vid(yt, output, start, end):
    url_base='https://www.youtube.com/watch?v='
    command = ['/Users/juxiang/opt/anaconda3/bin/youtube-dl',#'youtube-dl',
               #               '--quiet', '--no-warnings',
               '-f', 'bestvideo[ext=mp4]',
            #    '-o', '"%s"' % output,
               '-g', '"%s"' % (url_base + yt)] 
                # + '?start=' + str(start) + '&end=' + str(end))
    command = ' '.join(command)

    print(command)

    actual_url = os.popen(command).read()
    actual_url = actual_url.strip()

    # print(actual_url)

    command2 = ['/Users/juxiang/opt/anaconda3/bin/ffmpeg',
               '-ss', str(start-3),
               '-i', '"%s"' % (actual_url),
               '-t', str(end-start+2),
               '-c', 'copy ' + output + yt +'.mp4',
               '-an',
            #    '-filter:v "crop='+crop+'"',#600:460:300:190"',
               '-threads', '1',
               '-loglevel', 'panic']
    command2 = ' '.join(command2)

    # print(command2)

    os.system(command2)
                                                                    
# def wrapper(pitch):
#     local_clip('/ssd/ajpiergi/baseball/videos/'+file_.split('.')[0]+'.mkv', pitch['start']/fps, pitch['duration']/fps, pitch['clip_name']+'.mp4')
#     return 0
    
yt_ids = []
starts = []
ends = []
for p in data['pitch_data']:
    yt_ids.append(p['yt'])
    starts.append(p['start'])
    end = p['start'] + p['duration']
    # print(p['start'])
    # print(p['duration'])
    # print(end)
    ends.append(p['start']+p['duration'])

yt_ids = set(yt_ids)
print(yt_ids, len(yt_ids))

vid = '/Users/juxiang/Documents/InjuryVideo/Data/'
count = 0
for yt in yt_ids:
    # download yt video if doesnt exist
    if not os.path.exists(os.path.join(vid, yt+'.mp4')):
        down_vid(yt, vid, starts[count], ends[count])
    count += 1

exit()


# cmds = ['/home/ajpiergi/anaconda2/bin/youtube-dl -f bestvideo[ext=mp4] -o "/ssd/ptin/vid/jD86dDAbMMg.mp4" "https://www.youtube.com/watch?v=jD86dDAbMMg"',
#         '/home/ajpiergi/anaconda2/bin/youtube-dl -f bestvideo[ext=mp4] -o "/ssd/ptin/vid/tQkeOv-sJ3w.mp4" "https://www.youtube.com/watch?v=tQkeOv-sJ3w"',
#         '/home/ajpiergi/anaconda2/bin/youtube-dl -f bestvideo[ext=mp4] -o "/ssd/ptin/vid/jVsLXDA6BLs.mp4" "https://www.youtube.com/watch?v=jVsLXDA6BLs"',
#         '/home/ajpiergi/anaconda2/bin/youtube-dl -f bestvideo[ext=mp4] -o "/ssd/ptin/vid/zMa93aktzEY.mp4" "https://www.youtube.com/watch?v=zMa93aktzEY"',
#         '/home/ajpiergi/anaconda2/bin/youtube-dl -f bestvideo[ext=mp4] -o "/ssd/ptin/vid/WQeMos2_mko.mp4" "https://www.youtube.com/watch?v=WQeMos2_mko"',
#         '/home/ajpiergi/anaconda2/bin/youtube-dl -f bestvideo[ext=mp4] -o "/ssd/ptin/vid/R3dXKh5X9Oo.mp4" "https://www.youtube.com/watch?v=R3dXKh5X9Oo"',
#         '/home/ajpiergi/anaconda2/bin/youtube-dl -f bestvideo[ext=mp4] -o "/ssd/ptin/vid/cylP1wYXag4.mp4" "https://www.youtube.com/watch?v=cylP1wYXag4"',
#         '/home/ajpiergi/anaconda2/bin/youtube-dl -f bestvideo[ext=mp4] -o "/ssd/ptin/vid/gNvX_CMmOd4.mp4" "https://www.youtube.com/watch?v=gNvX_CMmOd4"',
#         '/home/ajpiergi/anaconda2/bin/youtube-dl -f bestvideo[ext=mp4] -o "/ssd/ptin/vid/F7PWPw1fYiE.mp4" "https://www.youtube.com/watch?v=F7PWPw1fYiE"',
#         '/home/ajpiergi/anaconda2/bin/youtube-dl -f bestvideo[ext=mp4] -o "/ssd/ptin/vid/_ZKFVvzIHGA.mp4" "https://www.youtube.com/watch?v=_ZKFVvzIHGA"',
#         '/home/ajpiergi/anaconda2/bin/youtube-dl -f bestvideo[ext=mp4] -o "/ssd/ptin/vid/I0jNhfAZs-U.mp4" "https://www.youtube.com/watch?v=I0jNhfAZs-U"',
#         '/home/ajpiergi/anaconda2/bin/youtube-dl -f bestvideo[ext=mp4] -o "/ssd/ptin/vid/GFDUgq0ae80.mp4" "https://www.youtube.com/watch?v=GFDUgq0ae80"',
#         '/home/ajpiergi/anaconda2/bin/youtube-dl -f bestvideo[ext=mp4] -o "/ssd/ptin/vid/Os7i4IOAUBI.mp4" "https://www.youtube.com/watch?v=Os7i4IOAUBI"',
#         '/home/ajpiergi/anaconda2/bin/youtube-dl -f bestvideo[ext=mp4] -o "/ssd/ptin/vid/AWmTrJItceA.mp4" "https://www.youtube.com/watch?v=AWmTrJItceA"',
#         '/home/ajpiergi/anaconda2/bin/youtube-dl -f bestvideo[ext=mp4] -o "/ssd/ptin/vid/eA-Jooav_UA.mp4" "https://www.youtube.com/watch?v=eA-Jooav_UA"',
#         '/home/ajpiergi/anaconda2/bin/youtube-dl -f bestvideo[ext=mp4] -o "/ssd/ptin/vid/gnEW5UZPYTY.mp4" "https://www.youtube.com/watch?v=gnEW5UZPYTY"',
#         '/home/ajpiergi/anaconda2/bin/youtube-dl -f bestvideo[ext=mp4] -o "/ssd/ptin/vid/H-Iz_SzdXls.mp4" "https://www.youtube.com/watch?v=H-Iz_SzdXls"',
#         '/home/ajpiergi/anaconda2/bin/youtube-dl -f bestvideo[ext=mp4] -o "/ssd/ptin/vid/gE-C6jtKV4Y.mp4" "https://www.youtube.com/watch?v=gE-C6jtKV4Y"',
#         '/home/ajpiergi/anaconda2/bin/youtube-dl -f bestvideo[ext=mp4] -o "/ssd/ptin/vid/i9C83IubIw8.mp4" "https://www.youtube.com/watch?v=i9C83IubIw8"',
#         '/home/ajpiergi/anaconda2/bin/youtube-dl -f bestvideo[ext=mp4] -o "/ssd/ptin/vid/vx4ooBUXdlA.mp4" "https://www.youtube.com/watch?v=vx4ooBUXdlA"',
#         '/home/ajpiergi/anaconda2/bin/youtube-dl -f bestvideo[ext=mp4] -o "/ssd/ptin/vid/OrTuLYhmuQQ.mp4" "https://www.youtube.com/watch?v=OrTuLYhmuQQ"',
#         '/home/ajpiergi/anaconda2/bin/youtube-dl -f bestvideo[ext=mp4] -o "/ssd/ptin/vid/a7paOawCa-I.mp4" "https://www.youtube.com/watch?v=a7paOawCa-I"',
#         '/home/ajpiergi/anaconda2/bin/youtube-dl -f bestvideo[ext=mp4] -o "/ssd/ptin/vid/dNczsKnLHWg.mp4" "https://www.youtube.com/watch?v=dNczsKnLHWg"',
#         '/home/ajpiergi/anaconda2/bin/youtube-dl -f bestvideo[ext=mp4] -o "/ssd/ptin/vid/h2HcdcpxYyE.mp4" "https://www.youtube.com/watch?v=h2HcdcpxYyE"',
#         '/home/ajpiergi/anaconda2/bin/youtube-dl -f bestvideo[ext=mp4] -o "/ssd/ptin/vid/yVsspUDDHC0.mp4" "https://www.youtube.com/watch?v=yVsspUDDHC0"',
#         '/home/ajpiergi/anaconda2/bin/youtube-dl -f bestvideo[ext=mp4] -o "/ssd/ptin/vid/AdAgo1spHho.mp4" "https://www.youtube.com/watch?v=AdAgo1spHho"',
#         '/home/ajpiergi/anaconda2/bin/youtube-dl -f bestvideo[ext=mp4] -o "/ssd/ptin/vid/irLSdOmwAyg.mp4" "https://www.youtube.com/watch?v=irLSdOmwAyg"',
#         '/home/ajpiergi/anaconda2/bin/youtube-dl -f bestvideo[ext=mp4] -o "/ssd/ptin/vid/x6vSIx9aF4I.mp4" "https://www.youtube.com/watch?v=x6vSIx9aF4I"']

# def run(cmd):
#    os.system(cmd)
# pool = multiprocessing.Pool(processes=8)
# pool.map(run, cmds)
   
# exit()
# for i in [0]:
#     pool = multiprocessing.Pool(processes=8)
#     pool.map(wrapper, data['game_data'])
    
