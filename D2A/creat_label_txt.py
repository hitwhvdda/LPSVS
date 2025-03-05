import os
import json
import linecache
import time

projectlist=["ffmpeg","httpd","libav","nginx","openssl"]
#projectlist=["httpd"]
leftnumlist=[38,37,36,36,40]
DEST_FILE = "/home/ao_ding/gwork/data/D2A/data_a/"

bug_type_list=[]
rootcwd="/home/ao_ding/gwork/data/D2A/data_a/"
for project,leftnum in zip(projectlist,leftnumlist):
    workcwd = rootcwd+"/"+project+"_trace"
    os.chdir(workcwd)
    filelist=os.listdir()
    for file in filelist:
        filepath=workcwd+"/"+file
        if file[-4:]!="json":
            continue
        with open(filepath) as sf:
            file_content = json.load(sf)
            id1,id2,_= file_content["id"].split("_")
            id = id1+"_"+id2
            bug_type=file_content["bug_type"]
            bug_type_list.append(bug_type)
            traces=file_content["trace"]
            labelfile = DEST_FILE+project+"_label/"+id+"-"+bug_type+".txt"
            lf=open(labelfile,mode= 'w')
            print("begin read trace")
            havefile=True
            cachelist=[]
            for trace in traces:
                filename=""
                location = trace["url"]
                url,line = location.split("/#L")
                filenamelist=url[leftnum:].split('/')
                length=len(filenamelist)
                for i in range(length-1):
                    filename=filename+filenamelist[i]+'_'
                last=filenamelist[length-1].replace('\n',' ')
                filename=filename+last
                print(filename,line)
                if(os.path.exists(filename)==False):
                    havefile=False
                    continue
                cache=linecache.getline(filename,int(line))
                print("cache",cache)
                cachelist.append(cache)
            if(havefile==True):
                print("creat a file",labelfile)
                lf.writelines(cachelist)
                lf.close()
            else:
                print("delete a file",labelfile)
                lf.close()
                os.system("rm %s"%labelfile)
        

bugtypelist=set(bug_type_list)
for bugtype in bugtypelist:
    print('The number ', bugtype , ' is ' ,str(bug_type_list.count(bugtype)))
