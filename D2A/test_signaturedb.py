import os
import sqlite3

label_idx=["Buffer_overflow","Integer_overflow","Alloc_may_be_big","Null_dereference","unitialized_value","Dead_store"]
def selectall(labelnum=0):
    conn=sqlite3.connect(r"/home/ao_ding/expand/D2A/database/CodeBERT_BGRU.db")
    c=conn.cursor()
    c.execute("select * from signatures where label=%d"%labelnum)
    signature_list=c.fetchall()
    print("type:",label_idx[labelnum])
    print(len(signature_list))
    for i in range(5):
        print(signature_list[i])
    conn.close()

def deleteall():
    conn=sqlite3.connect(r"/home/ao_ding/expand/D2A/database/CodeBERT_BGRU.db")
    c=conn.cursor()
    c.execute("delete from signatures")
    conn.commit()
    conn.close()


#selectall()
#deleteall()
for i in range(6):
    selectall(i)
