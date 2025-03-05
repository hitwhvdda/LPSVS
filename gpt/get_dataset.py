import os
from sklearn.model_selection import train_test_split

number = [0,0,0,0,0,0]
vullist = ['IntOverflow','IntUnderflow','CmdInjection','DoubleFree','UseAfterFree','FormatString']
def read_dataset():
    rootcwd = "/home/ao_ding/expand/trace_gptslice/data/GPT_slice/after_process/"
    dirlist = os.listdir(rootcwd)
    datalist = []
    labellist = []
    filenamelist = []
    for dir in dirlist:
        if("IntOverflow" in dir):
            label = 0
        if("IntUnderflow" in dir):
            label = 1
        if("CmdInjection" in dir):
            label = 2
        if("DoubleFree" in dir):
            label = 3
        if("UseAfterFree" in dir):
            label = 4
        if("FormatString" in dir):
            label = 5
        workdir = rootcwd +dir
        filelist = os.listdir(workdir)
        for filename in filelist:
            filepath = workdir + "/" + filename
            #print(labeldir)
            a_datalist = []
            f = open(filepath,'r')
            line = f.readline()
            if(line.find('\n')>1):
                line=line.replace('\n','')
            while line:
                a_datalist.append(line)
                line = f.readline()
                if(line.find('\n')>1):
                    line=line.replace('\n','')
            f.close()
            if(len(a_datalist)==0):
                continue
            datalist.append(a_datalist)
            labellist.append(label)
            filenamelist.append(filename)
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



#dataset_list = all_dataset(0.8,0.1,0.1)
#print(dataset_list[:5])