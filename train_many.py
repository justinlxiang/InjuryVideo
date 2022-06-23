import time
import multiprocessing
import itertools
import os
import random

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

k = 20
exps = [(name, name2, k) for name,name2 in itertools.product(names,names)]
random.shuffle(exps)
exps = [(i,e) for i,e in enumerate(exps[:36])]

#for name, k in exps:
def run(e):
    i,e = e
    name, name2, k = e
    cmd = "python train_model.py -model_file=t"+str(i)+" -gpu="+str(i%4)+" -name='"+name+"' -name2='"+name2+"' -k="+str(k)+" > logs_double/"+name.replace(" ","_").replace(".","")+"_"+str(name2).replace(' ','_').replace('.','')+".txt"
#    os.system(cmd)
    print cmd
#    time.sleep(5)
    os.system(cmd)

p = multiprocessing.Pool(4)
p.map(run, exps)
