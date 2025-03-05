import get_dataset as gds


dataset = gds.all_dataset(0.8,0.1,0.1)
print(len(dataset))
f = open('/home/ao_ding/expand/SySeVR/code/corpus.txt',"w")
for a_data in dataset:
        
    print("a_data:",a_data)
    slice = a_data[0]
    print("slice:",slice)
    for a_slice in slice:
        print(a_slice)
        f.write(a_slice+'\n')
f.close()
            
print("write success")
# max=0
# for a_data in dataset:
#     slice = a_data[0]
#     length = len(slice)
#     if length>max:
#         max = length

# print(max)