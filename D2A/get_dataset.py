import os
from sklearn.model_selection import train_test_split

rootcwd="/home/ao_ding/gwork/data/D2A/data_a/"
project_list=["httpd","libav","openssl","nginx","ffmpeg"]
def read_dataset():
    datalist = []
    labellist = []
    filenamelist = []
    for project in project_list:
        workcwd=rootcwd+project+"_label/"
        print("workcwd:",workcwd)
        os.chdir(workcwd)
        filelist = os.listdir(workcwd)
        for labelfile in filelist:
            #print("labelfile:",labelfile)
            if(labelfile.find("txt")<0 and labelfile=="all_length.txt"):
                continue
            workfile = workcwd + "//" +labelfile
            
            filename = labelfile.split('.')[0]
            #print("filename:",filename)
            label=filename.split("-")[-1]
            #print("label:",label)
            if(label>"5"):
                continue
            a_datalist = []
            f = open(labelfile)
            line = f.readline()
            if(line.find('\n')>1):
                line=line.replace('\n','')
            while line:
                a_datalist.append(line)
                line = f.readline()
                if(line.find('\n')>1):
                    line=line.replace('\n','')
            datalist.append(a_datalist)
            filenamelist.append(labelfile)
            #labellist.append(int(label))
            labellist.append(label)
            f.close()
                
    return datalist,labellist,filenamelist

def all_dataset(train_rat,valid_rat,test_rat,split = True,average_set = True):
    datalist,labellist,filenamelist = read_dataset()
    newdatalist=[]
    for data,filename in zip(datalist,filenamelist):
        newdatalist.append([data,filename])
    dataset_list = []
    if average_set ==True:
        stratify = labellist
    else:
        stratify = None
    x_train,x_test,y_train,y_test = train_test_split(
        newdatalist,labellist,test_size=1-train_rat,random_state=1,stratify=stratify
    )
    if average_set ==True:
        stratify = y_test
    else:
        stratify = None
    x_valid,x_test,y_valid,y_test = train_test_split(
        x_test,y_test,test_size=test_rat / (test_rat + valid_rat),random_state=1,stratify=stratify
    )
    
    for data,label in zip(x_train,y_train):
        dataset_list.append((data[0],int(label),0))
    for data,label in zip(x_valid,y_valid):
        dataset_list.append((data[0],int(label),1))
    for data,label in zip(x_test,y_test):
        dataset_list.append((data[0],int(label),2))
    return dataset_list

