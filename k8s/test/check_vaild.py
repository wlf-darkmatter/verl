import pandas as pd
import numpy as np
import itertools

list_total_ip=[
"10.127.44.182",
"10.127.44.202",
"10.127.44.185",
"10.127.44.192",
"10.127.44.191",
"10.127.44.195",
"10.127.44.193",
"10.127.44.197",
"10.127.18.178",
"10.127.18.180",
"10.127.18.162",
"10.127.18.167",
"10.127.18.176",
"10.127.18.185",
"10.127.18.165",
"10.127.18.182",
"10.127.18.163",
"10.127.18.183",
"10.127.44.201",
"10.127.44.188",
"10.127.44.200",
"10.127.44.198",
"10.127.44.199",
"10.127.44.183",
"10.127.44.194",
"10.127.44.186",
"10.127.44.196",
"10.127.44.187",
"10.127.44.190",
"10.127.44.189",
"10.127.44.184",
"10.127.18.170",
"10.127.18.187",
"10.127.18.179",
"10.127.18.184",
"10.127.18.174",
"10.127.18.177",
"10.127.18.172",
"10.127.18.175",
"10.127.18.169",
"10.127.18.188",
"10.127.18.164",
"10.127.18.181",
"10.127.18.166",
"10.127.18.186",
"10.127.18.173",
"10.127.18.171",
"10.127.18.168",
]
list_OK_ip="10.127.44.186 10.127.44.201 10.127.44.199 10.127.44.200 10.127.44.193 10.127.44.198 10.127.44.197 10.127.44.202 10.127.18.164 10.127.18.169 10.127.18.171 10.127.18.186 10.127.18.174 10.127.18.175 10.127.18.162 10.127.18.184 10.127.18.185 10.127.18.188"
list_OK_ip=list_OK_ip.split(" ")

np_list_notOK_ip="10.127.44.187 10.127.44.182 10.127.44.192 10.127.44.184 10.127.44.189 10.127.44.188 10.127.44.195 10.127.44.190 10.127.44.196 10.127.44.191 10.127.44.194 10.127.44.185 10.127.44.183 10.127.18.181 10.127.18.179 10.127.18.178 10.127.18.173 10.127.18.170 10.127.18.182 10.127.18.177 10.127.18.187 10.127.18.167 10.127.18.166 10.127.18.165 10.127.18.172 10.127.18.163 10.127.18.183 10.127.18.180 10.127.18.176 10.127.18.168"
np_list_notOK_ip=np_list_notOK_ip.split(" ")


n_group=2
np_list_OK_ip=np.array(list_OK_ip).reshape(-1, n_group)
np_list_notOK_ip=np.array(np_list_notOK_ip).reshape(-1, n_group)
np_rank_table = np.zeros((len(list_total_ip), len(list_total_ip)), dtype=np.object_)
for i in range(len(list_total_ip)):
    for j in range(len(list_total_ip)):
        np_rank_table[i,j] = ""

for group in np_list_OK_ip:
    print(group)
    list_index = [list_total_ip.index(i) for i in group]
    print(list_index)
    for a,b in itertools.combinations(list_index, 2):
        np_rank_table[a,b]= "√"
        np_rank_table[b,a]= "√"

for group in np_list_notOK_ip:
    print(group)
    list_index = [list_total_ip.index(i) for i in group]
    print(list_index)
    for a,b in itertools.combinations(list_index, 2):
        np_rank_table[a,b]= "x"
        np_rank_table[b,a]= "x"


df = pd.DataFrame(np_rank_table)
df.to_csv("rank_table.csv")
df.to_excel("rank_table.xlsx")