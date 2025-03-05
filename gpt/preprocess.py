import os
import re

rootdir = "/home/ao_ding/expand/trace_gptslice/data/GPT_slice/first_extract/"
outputdir = "/home/ao_ding/expand/trace_gptslice/data/GPT_slice/after_process/"
wrongfilelist = []
vullist = ['IntOverflow','IntUnderflow','CmdInjection','DoubleFree','UseAfterFree','FormatString']
def process_code(input_file_path, output_file_path):
    with open(input_file_path, 'r') as file:
        content = file.read()

    # 去除多行注释
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
    # 去除单行注释
    content = re.sub(r'//.*', '', content)
    # 去除空行和只包含大括号的行
    content = re.sub(r'^\s*[{}]\s*$', '', content, flags=re.MULTILINE)

    # 将处理后的内容按行分割
    lines = content.split('\n')
    processed_lines = []
    current_line = ''

    for line in lines:
        line = line.strip()
        # 如果当前行为空或只包含大括号，则跳过
        if not line or re.match(r'^\s*[{}]\s*$', line) or re.match(r'^\s*\.\.\.\s*$', line):
            continue
        if line.lstrip().startswith('if'):
            processed_lines.append(line)
            continue
        if line.lstrip().startswith('#'):
            continue
        # 如果当前行不以分号结尾，则需要与下一行合并
        if not line.endswith(';'):
            if current_line:
                current_line += ' ' + line
            else:
                current_line = line
        else:
            if current_line:
                processed_lines.append(current_line + ' ' + line)
                current_line = ''
            else:
                processed_lines.append(line)

    # 将处理后的代码写入新文件
    with open(output_file_path, 'w') as file:
        for line in processed_lines:
            file.write(line + '\n')

for vul in vullist:
    workdir = rootdir + vul
    readfilelist = os.listdir(workdir)
    for readfile in readfilelist:
        process_code(workdir+"/"+readfile,outputdir+vul+"/"+readfile)
