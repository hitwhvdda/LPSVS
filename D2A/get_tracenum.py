import os

dirlist=["ffmpeg_trace","httpd_trace","openssl_trace","libav_trace","libtiff_trace","nginx_trace"]

for a_dir in dirlist:
    dest_file="/home/ao_ding/gwork/data/D2A/"+a_dir
    os.chdir(dest_file)
    length=len(os.listdir())
    print(a_dir,length)

