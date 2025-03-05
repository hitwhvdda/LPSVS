import os
from sklearn.model_selection import train_test_split

number = [0,0,0,0]
def read_dataset():
    rootcwd = "/home/ao_ding/expand/SySeVR/data/slice_label_vul/"
    dirlist = os.listdir(rootcwd)
    datalist = []
    labellist = []
    filenamelist = []
    for dir in dirlist:
        workdir = rootcwd +dir
        labeldirlist = os.listdir(workdir)
        for labelname in labeldirlist:
            labeldir = workdir + "/" + labelname
            #print(labeldir)
            if(os.path.isdir(labeldir)):
                labelfile = os.listdir(labeldir)[0]
                #print(labelfile)
                label,_ = labelfile.split('.')
                a_datalist = []
                f = open(labeldir + "/" + labelfile)
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
                filenamelist.append(labelname)
    return datalist,labellist,filenamelist

def all_dataset(train_rat,valid_rat,test_rat,split = True,average_set = True):
    datalist,labellist,filenamelist = read_dataset()
    API_list = ['CWE23','CWE126','CWE127','CWE194','CWE195']
    Arithmetic_list = ['CWE190','CWE369','CWE680']
    array_list = ['CWE121','CWE122','CWE590']
    pointer_list = ['CWE36','CWE78','CWE124','CWE134','CWE789']
    selectdata = []
    selectlabel = []
    for data,label,filename in zip(datalist,labellist,filenamelist):
        CWEtype = filename.split("_")[2]
        #print("filename",filename)
        #print("CWETYPE",CWEtype)
        worklist = []
        if label == 0:
            worklist = API_list
        elif label == 1:
            worklist = Arithmetic_list
        elif label == 2:
            worklist = array_list
        elif label == 3:
            worklist = pointer_list
        if CWEtype in worklist:
            selectdata.append(data)
            selectlabel.append(label)
            number[label] = number[label] + 1
    dataset_list = []
    print(number)
    if average_set ==True:
        stratify = selectlabel
    else:
        stratify = None
    x_train,x_test,y_train,y_test = train_test_split(
        selectdata,selectlabel,test_size=1-train_rat,random_state=1,stratify=stratify
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


#dataset_list = all_dataset(0.8,0.1,0.1)
#print(dataset_list[:5])