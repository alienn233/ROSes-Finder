import numpy as np
import pandas as pd
import sys
args=sys.argv


a3 = pd.read_csv(args[1],header=0,sep=",")
tp = 0
tn = 0
fn = 0
fp = 0
for j in range(0,999):
    if int(a3["c"][j])==1 and int(a3["id"][j])==1:
        tp=tp+1
    if int(a3["c"][j])==1 and int(a3["id"][j])!=1:
        fn=fn+1
    if int(a3["c"][j])!=1 and int(a3["id"][j])==1:
        fp=fp+1
    if int(a3["c"][j])==1 and int(a3["id"][j])!=1:
        tn=tn+1  
acc= 1. *  (tp+tn)/(tp+tn+fp+fn)
ecall=tp/(tp+fn)
precision = tp / (tp + fp)
sensitive =  tp / (tp + fn)
f1 = 2 * precision * sensitive / (precision + sensitive)
print(acc,ecall,precision,sensitive,f1)
#print(ecall)
#print(precision)
#print(sensitive)
#print(f1,acc)
