from openai import OpenAI
import os
import sys

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key="sk-pgwBcgZsbhGFDTmtL9WJULceWDGUbAmqIaeCxqhnDKFOhQix",
    base_url="https://api.chatanywhere.tech/v1"
)

vullist = ['BufferOverflow','IntOverflow','IntUnderflow','CmdInjection','DoubleFree','UseAfterFree','FormatString']
datadir = '/home/ao_ding/expand/trace_gptslice/data/source_code/'
gptoutdir = '/home/ao_ding/expand/trace_gptslice/data/gpt_output/'

def combine_files_with_reference(file_paths):
    """将多个文件内容合并，并标注引用关系

    Args:
        file_paths (list): 文件路径列表
    Returns:
        str: 合并后的文件内容
    """
    combined_content = ""
    for i, file_path in enumerate(file_paths):
        with open(file_path, 'r') as file:
            file_content = file.read()
            # 在合并时标注文件名和内容
            combined_content += f"\n# File {i+1}: {os.path.basename(file_path)}\n"
            combined_content += file_content + "\n"
    return combined_content

def gpt_4_api_stream(messages: list):
    """为提供的对话消息创建新的回答 (流式传输)

    Args:
        messages (list): 完整的对话消息
    """
    stream = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=messages,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="")

if __name__ == '__main__':
    vul_type = 6
    sample_prompt = "This is an example to give some source code of a programme containing API_function_call type vulnerability as follows\n:"
    sampledir = "/home/ao_ding/expand/trace_gptslice/data/testdata/"
    samplepath = sampledir + vullist[vul_type] + "/sample/"
    samplelist = os.listdir(samplepath)
    sample_file_list = []
    for samplefile in samplelist:
        sample_file_list.append(samplepath + samplefile)
    sample_file_content = combine_files_with_reference(sample_file_list)
    sample_prompt = sample_prompt + sample_file_content + "\n For the source code, based on the data dependency and control dependency of the program, we can find the key code statements related to the vulnerability triggering, and generate the following vulnerability slices:\n"
    sample_slice_flie = sampledir + vullist[vul_type] + "/slice.txt"
    with open(sample_slice_flie, 'r') as file:
        sample_slice = file.read()
    
    sample_prompt = sample_prompt + sample_slice
    count = 0
    rootcwd = datadir + vullist[vul_type]
    dirlist = os.listdir(rootcwd)
    for dir in dirlist:
        print("For:",dir)
        workdir=os.path.join(rootcwd,dir)
        filelist = os.listdir(workdir)
        file_path_list = []
        for filename in filelist:
            file_path = workdir + '/' + filename 
            file_path_list.append(file_path)
        file_content = combine_files_with_reference(file_path_list)
        prompt = f"\nPlease perform a vulnerability check on the following code file based on the example above to detect a vulnerability related to {vullist[vul_type]} and give a code slice that triggers a highly relevant vulnerability trigger from data source point to vulnerability trigger point based on the program's data dependencies and program dependencies, starting with 'Begin' and ending with 'End',Note that the generated slices should not include statements that are not relevant to the code, such as comments.Make sure the code slice includes only statements that are highly relevant to the vulnerability:\n"
        gpt_input = sample_prompt + prompt + file_content
            #print(gpt_input)
        out_filename = gptoutdir + vullist[vul_type] + "/" + dir + '.txt'
        if(os.path.exists(out_filename)==True):
            continue
        with open(out_filename, 'w') as f:
            sys.stdout = f
            messages = [{'role': 'user','content': gpt_input},]
            gpt_4_api_stream(messages)
            
        sys.stdout = sys.__stdout__
        print("analysis" + dir + "success!")
        count = count + 1
    print("count:",count)