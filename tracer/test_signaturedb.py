import os
import sqlite3

label_idx=["CmdInjection.","FormatString.","IntOverflow.","IntUnderflow.","DoubleFree.","UseAfterFree.","BufferOverflow."]
def selectall(labelnum=0):
    conn=sqlite3.connect(r"/home/ao_ding/expand/trace/database/signature_Glove_BiLSTM.db")
    c=conn.cursor()
    c.execute("select * from signatures where label=%d"%labelnum)
    signature_list=c.fetchall()
    print("type:",label_idx[labelnum])
    for i in range(5):
        print(signature_list[i])
    conn.close()

def deleteall():
    conn=sqlite3.connect(r"/home/ao_ding/expand/trace/database/signature_Glove_BiLSTM.db")
    c=conn.cursor()
    c.execute("delete from signatures")
    conn.commit()
    conn.close()


#selectall()
#deleteall()
for i in range(7):
    selectall(i)
