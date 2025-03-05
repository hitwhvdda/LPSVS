import os
from sklearn.model_selection import train_test_split

def read_dataset():
    nowcwd = os.getcwd()
    if(nowcwd[-12:-1]!="signature-d"):
        rootcwd = nowcwd[:-4] + "signature-db"
        print("rootcwd",rootcwd)
        os.chdir(rootcwd)
    else:
        rootcwd = nowcwd
    dirlist = os.listdir(rootcwd)
    datalist = []
    labellist = []
    filenamelist = []
    for dir in dirlist:
        if(os.path.isdir(dir)==False or dir=='lightning_logs'):
            continue
        workdir = rootcwd + "//" +dir
        labeldirlist = os.listdir(workdir)
        for labeldir in labeldirlist:
            labeldir = workdir + "//" + labeldir
            #print(labeldir)
            if(os.path.isdir(labeldir)):
                labelfile = os.listdir(labeldir)[0]
                #print(labelfile)
                label,_ = labelfile.split('.')
                if(label == '7'):
                    continue
                a_datalist = []
                f = open(labeldir + "//" + labelfile)
                line = f.readline()
                if(line.find('\n')>1):
                    line=line.replace('\n','')
                while line:
                    a_datalist.append(line)
                    line = f.readline()
                    if(line.find('\n')>1):
                        line=line.replace('\n','')
                f.close()
                datalist.append(a_datalist)
                labellist.append(int(label))
                filenamelist.append(workdir)
    return datalist,labellist,filenamelist

def all_dataset(train_rat,valid_rat,test_rat,split = True,average_set = True):
    datalist,labellist,_ = read_dataset()
    dataset_list = []
    if average_set ==True:
        stratify = labellist
    else:
        stratify = None
    x_train,x_test,y_train,y_test = train_test_split(
        datalist,labellist,test_size=1-train_rat,random_state=1,stratify=stratify
    )
    if average_set ==True:
        stratify = y_test
    else:
        stratify = None
    x_valid,x_test,y_valid,y_test = train_test_split(
        x_test,y_test,test_size=test_rat / (test_rat + valid_rat),random_state=1,stratify=stratify
    )
    for data,label in zip(x_train,y_train):
        dataset_list.append((data,label,0))
    for data,label in zip(x_valid,y_valid):
        dataset_list.append((data,label,1))
    for data,label in zip(x_test,y_test):
        dataset_list.append((data,label,2))
    return dataset_list

