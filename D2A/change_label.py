import os
import json
import linecache
import time

projectlist=["ffmpeg","httpd","libav","nginx","openssl"]
ROOT_PATH = "/home/ao_ding/gwork/data/D2A/data_a/"
numlist=[0,0,0,0,0,0]

for project in projectlist:
    workcwd=ROOT_PATH+project+"_label/"
    os.chdir(workcwd)
    filelist=os.listdir()
    for files in filelist:
        left,types=files.split("-")
        types,_=types.split(".")
        if(types[:14]=="BUFFER_OVERRUN"):
            numlist[0]=numlist[0]+1
            newfile=left+"-0.txt"
        elif(types[:16]=="INTEGER_OVERFLOW"):
            numlist[1]=numlist[1]+1
            newfile=left+"-1.txt"
        elif(types[-3:]=="BIG"):
            numlist[2]=numlist[2]+1
            newfile=left+"-2.txt"
        elif(types[:4]=="NULL"):
            numlist[3]=numlist[3]+1
            newfile=left+"-3.txt"
        elif(types=="UNINITIALIZED_VALUE"):
            numlist[4]=numlist[4]+1
            newfile=left+"-4.txt"
        elif(types=="DEAD_STORE"):
            numlist[5]=numlist[5]+1
            newfile=left+"-5.txt"
        else:
            os.system("rm %s"%files)
            print("delete ",files)
            continue
        os.system("mv %s %s"%(files,newfile))
        print("change:",files," to:",newfile)


for i in range(6):
    print(numlist[i])
