import get_dataset as gds

dataset_list = gds.all_dataset(0.8,0.1,0.1)
max = 0
for data,_,_ in dataset_list:
    length = len(data)
    if(length >= max):
        max = length

print(max)