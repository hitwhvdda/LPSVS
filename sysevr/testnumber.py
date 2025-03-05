import os
import get_dataset as gds

rootdir = "/home/ao_ding/expand/SySeVR/data/select_data/"
#alist = ["API_function_call","Arithmetic_expression","Array_usage","Pointer_usage"]
numberlist = [0,0,0,0]
dir_list = os.listdir(rootdir)
for dir in dir_list:
    label=int(dir.split('_')[-1])
    numberlist[label] += 1

print(numberlist)    

# dataset_list = gds.all_dataset(0.8,0.1,0.1)
# numberlist = [0,0,0,0]
# for adata in dataset_list:
#     numberlist[int(adata[1])] = numberlist[int(adata[1])] + 1

# for i in range(len(alist)):
#     print(alist[i]+":"+str(numberlist[i]))