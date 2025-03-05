import os
import json


def creat_tabel():
    rootcwd=os.getcwd()
    workcwd = rootcwd+"/"+"httpd"+"_trace"
    os.chdir(workcwd)
    filelist=os.listdir()
    txtfile = workcwd + "/" + "url.txt"
    textfile = open(txtfile,"w")
    urllist=[]
    newfilelist=[]
    for file in filelist:
        filepath=workcwd+"/"+file
        if file[-4:]!="json":
            continue
        with open(filepath) as sf:
            file_content = json.load(sf)
            traces=file_content["trace"]
            for trace in traces:
                location = trace["url"]
                url,line = location.split("/#L")
                newfile=url.split("/")[-1]
                urllist.append(url)
                newfilelist.append(newfile)
                
            sf.close()
    urllist=list(set(urllist))
    for url in urllist:
        textfile.write(url+os.linesep)
    textfile.close()
    nf = open(workcwd+"/"+"newfile.txt","w")
    newfilelist.sort()
    for newfile in newfilelist:
        nf.write(newfile+os.linesep)
    nf.close()

creat_tabel()

