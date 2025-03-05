import os
import time
nowcwd=os.getcwd()
workcwd = nowcwd + "/nginx_" + "trace"
os.chdir(workcwd)
txtfile=workcwd+"/url.txt"
tf=open(txtfile,"r")
urllist=tf.readlines()
for url in urllist:
    filename=""
    a_list=url[36:].split("/")
    length = len(a_list)
    for i in range(length-1):
        filename=filename+a_list[i]+"_"
    a=a_list[length-1].replace('\n',' ')
    filename = filename+a
    if os.path.exists(filename)==True:
        continue
    print(filename)
    url = "https://raw.fastgit.org/nginx/nginx"+url[35:]
    cmd = "wget -O "+filename + " " + url
    print(cmd)
    os.system(cmd)
    time.sleep(10)
