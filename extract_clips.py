import os
import json
import string
import random

import subprocess
import multiprocessing

with open('injury_data.json', 'r') as f:
        data = json.load(f)

crops = {'d':'600:460:300:190',
         'vx4ooBUXdlA': '600:460:200:190',
         'H-Iz_SzdXls': '600:460:140:190',
         'F7PWPw1fYiE':'600:460:100:190',
         'AdAgo1spHho': '600:460:180:180',
         'dNczsKnLHWg':'600:460:230:190',
         'AWmTrJItceA':'600:460:280:190',
         'EuvlwwrqAtA':'600:460:240:190',
         'zMa93aktzEY':'600:460:230:190',
         'jD86dDAbMMg':'600:460:240:190',
         'dNczsKnLHWg': '600:460:230:190',
         'gnEW5UZPYTY': '600:460:250:190',
         'XOdpZqbH58k': '600:460:310:200',
         'tQkeOv-sJ3w': '600:460:260:200',
         'I0jNhfAZs-U': '600:460:280:200',
         'GFDUgq0ae80': '600:460:280:200',
         'eA-Jooav_UA': '600:460:260:200',
         '_ZKFVvzIHGA': '600:460:290:200',
         'R3dXKh5X9Oo': '600:460:210:200',
         'AdAgo1spHho': '600:460:260:200',
         'WQeMos2_mko':'600:460:220:200',
         'AWmTrJItceA': '600:460:210:200',
         'Os7i4IOAUBI': '600:460:280:200',
         'a7paOawCa-I': '600:460:310:190',
}
def local_clip(yt, start_time, duration, output_filename):
    url_base='https://www.youtube.com/watch?v='
    command = ['/Users/juxiang/opt/anaconda3/bin/youtube-dl',#'youtube-dl',
               #               '--quiet', '--no-warnings',
               '-f', 'bestvideo[ext=mp4]',
         #    '-o', '"%s"' % output,
               '-g', '"%s"' % (url_base + yt)] 
                # + '?start=' + str(start) + '&end=' + str(end))
    command = ' '.join(command)
    
    print(output_filename)
    # print(command)

    actual_url = os.popen(command).read()
    actual_url = actual_url.strip()

    # print(actual_url)

    end_time = start_time + duration
    vid = yt
    crop = crops[vid] if vid in crops else crops['d']
    #print vid
    #return 0
    command2 = ['/Users/juxiang/opt/anaconda3/bin/ffmpeg',
               '-ss', str(start_time),
               '-i', '"%s"' % actual_url,
               '-t', str(end_time - start_time),
               '-c copy', '/Users/juxiang/Documents/InjuryVideo/Data/%s' % (output_filename),
               #'-c:v', 'copy',
               '-an',
               '-filter:v "crop='+crop+'"',#600:460:300:190"',
               '-threads', '1',
               '-loglevel', 'panic']
    command2 = ' '.join(command2)

    # print(command2)
    os.system(command2)

    try:
        output = subprocess.check_output(command, shell=True,
                                         stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as err:
        print(err.output)
        return err.output


                                                                    
def wrapper(pitch):
    if not os.path.exists('/Users/juxiang/Documents/InjuryVideo/Data/'+pitch['clip_name']+'.mp4'):
        local_clip(pitch['yt'], pitch['start'], pitch['duration'], pitch['clip_name']+'.mp4')
        return 0

done = []
ptc = []
for d in data['pitch_data']:
    if d['yt'] not in done:
        done.append(d['yt'])
        ptc.append(d)
print(len(data['pitch_data']))
for i in [0]:
    if __name__ == '__main__':
        pool = multiprocessing.Pool(processes=8)
        # print(i)
        pool.map(wrapper, data['pitch_data'])
    
