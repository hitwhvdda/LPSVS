# CmdInjection.:0
# FormatString.:1
# IntOverflow.:2
# IntUnderflow.:3
# DoubleFree.:4
# UseAfterFree.:5
# BufferOverflow.:6
import os
import json
import linecache
import shutil

def rename(file):
    filename = file.split('.')
    os.rename(file,filename[0]+".txt")
    return filename[0]+".txt", filename[1]

def derename(file,last):
    filename = file.split('.')
    os.rename(file,filename[0]+"."+last)

count = [0,0,0,0,0,0,0]
if __name__ == '__main__':
    label_idx={"CmdInjection.":0,"FormatString.":1,"IntOverflow.":2,"IntUnderflow.":3,"DoubleFree.":4,"UseAfterFree.":5,"BufferOverflow.":6}
    rootcwd = os.getcwd()
    print("rootcwd:",rootcwd)
    dirs_ls=os.listdir(path=rootcwd)
    for dir in dirs_ls:
        print("dir:",dir)
        if(dir[0:6]=='buffer' or dir[0:7]=='command' or dir=='get_slice.py'):
            continue
        #if(dir[0:3]=='CWE'):
        workcwd = rootcwd + "/" +dir
        print("workcwd:",workcwd)
        file_ls=os.listdir(path=workcwd)
        os.chdir(workcwd)
        if(dir[0:3]=='CWE'):
            #Win
            #program_cwd = rootcwd[:-25]+"\\"+"tracer"
            #linux
            program_cwd = rootcwd[:-13]+"/"+"data"
            #print(program_cwd)
        else:
            #linux
            program_cwd = rootcwd[:-13]+"/data/infer-experiment/bench/"+dir

        print("program_cwd:",program_cwd)
        label_num = 0
        for file in file_ls:
            if (os.path.isdir(file)):
                shutil.rmtree(file)
            if (file.find("json")>1):
                with open(file=file) as f:
                    file_content = json.load(f)
                    type=label_idx[file_content["qualifier"]]
                    #print(type)
                    traces=file_content["bug_trace"]
                    for trace in traces:
                        a=0
                        count[type]+=1
                        lable_cwd = workcwd+ "/" + "label_" + str(label_num+a)
                        while(os.path.exists(lable_cwd) == True):
                            a=a+1
                            lable_cwd = workcwd+ "/" + "label_" + str(label_num+a)
                            
                            
                        lable_filepath = lable_cwd + "/" + str(type) + ".txt"
                        #if(os.path.exists(lable_filepath)):
                         #   os.remove(lable_filepath)
                        #if(os.path.exists(lable_cwd)):
                         #   os.rmdir(lable_cwd)
                        os.mkdir(lable_cwd)
                        lable_file = open(lable_filepath,mode= 'w')
                            
                        for sentence in trace:
                            #print(sentence)
                            file_name=sentence["filename"]
                            line_number=sentence["line_number"]
                            #print(file_name,line_number)
                            if(dir[0:3]=='CWE'):
                                dest_file = program_cwd + file_name[23:]
                            else:
                                dest_file = program_cwd + "/" + file_name
                            #dest_file = dest_file.replace('/','\\')
                            #dest_file,last=rename(dest_file)
                            #print("dest_file:",dest_file)
                            lable_file.writelines(linecache.getline(dest_file,line_number))
                            #derename(dest_file,last)

                        lable_file.close()
                        print("creat label_file:",lable_filepath)

        #elif(dir[0:6]!='buffer' and dir[0:7]!='command'):
            # workcwd = rootcwd + "\\" +dir
            # file_ls=os.listdir(path=workcwd)
            # os.chdir(workcwd)
            # #linux
            # program_cwd = rootcwd[:-13]+"\\data\\infer-experiment\\bench\\"+dir
            # label_num = 0
            # for file in file_ls:
            #     if (file.find("json")>1):
            #         with open(file=file) as f:
            #             file_content = json.load(f)
            #             type=label_idx[file_content["qualifier"]]
            #             #print(type)
            #             traces=file_content["bug_trace"]
            #             for trace in traces:
            #                 a=0
            #                 count[type]+=1
            #                 lable_cwd = workcwd+ "\\" + "label_" + str(label_num+a)
            #                 while(os.path.exists(lable_cwd) == True):
            #                     lable_cwd = workcwd+ "\\" + "label_" + str(label_num+a)
            #                     a=a+1
                            
            #                 lable_filepath = lable_cwd + "\\" + str(type) + ".txt"
            #                 if(os.path.exists(lable_filepath)):
            #                     os.remove(lable_filepath)
            #                 if(os.path.exists(lable_cwd)):
            #                     os.rmdir(lable_cwd)
            #                 os.mkdir(lable_cwd)
            #                 lable_file = open(lable_filepath,mode= 'w')

            #                 for sentence in trace:
            #                     #print(sentence)
            #                     file_name=sentence["filename"]
            #                     line_number=sentence["line_number"]
            #                     #print(file_name,line_number)
            #                     dest_file = program_cwd + "\\" + file_name
            #                     dest_file = dest_file.replace('/','\\')
            #                     dest_file,last=rename(dest_file)
            #                     #print(dest_file)
            #                     lable_file.writelines(linecache.getline(dest_file,line_number))
            #                     derename(dest_file,last)

            #                 lable_file.close()
            #                 print("creat label_file:",lable_filepath)
    keys = list(label_idx.keys())
    for i in range(7):
        print(keys[i],count[i])

