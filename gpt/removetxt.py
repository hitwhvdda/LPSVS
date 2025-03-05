import os

rootdir = "/home/ao_ding/expand/trace_gptslice/data/gpt_output/"
outputdir = "/home/ao_ding/expand/trace_gptslice/data/gpt_output_remove/"
wrongfilelist = []
vullist = ['IntOverflow','IntUnderflow','CmdInjection','DoubleFree','UseAfterFree','FormatString']
def extract_content(input_dir, output_dir,filename):
    file_path = os.path.join(input_dir, filename)
    output_file_path = os.path.join(output_dir, filename)
    file_in=open(file_path, 'r', encoding='utf-8')
    file_out=open(output_file_path, 'w', encoding='utf-8') 
    for line in file_in:
        # 检查指定字符是否存在于当前行
        if 'Begin' in line and 'End' in line:
            continue
            # 如果不存在，写入目标文件
        else:
            file_out.write(line)

for vul in vullist:
    workdir = rootdir + vul
    readfilelist = os.listdir(workdir)
    for readfile in readfilelist:
        extract_content(workdir,outputdir+vul,readfile)
