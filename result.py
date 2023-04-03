import pandas as pd

res=pd.read_csv('checkclass.csv')
output_acc=[]
output_recall=[]
data_acc={
}
data_recall={
}

for i in range(0,26):
    data_recall.clear()
    data_acc.clear()
    for k in range(1,4):
        tp=0
        tn=0
        fn=0
        fp=0
        for j in range(0,33748):
            if res.loc[j][k]==i and res.loc[j][4]==i:
                tp=tp+1
            if res.loc[j][k]==i and res.loc[j][4]!=i:
                fn=fn+1
            if res.loc[j][k]!=i and res.loc[j][4]==i:
                fp=fp+1
            if res.loc[j][k]!=i and res.loc[j][4]!=i:
                tn=tn+1
        acc=(tp+tn)/(tp+tn+fp+fn)
        recall=tp/(tp+fn)
        if k==1:
            data_acc['xgb']=acc
            data_recall['xgb']=recall
        if k==2:
            data_acc['nn']=acc
            data_recall['nn']=recall
        if k==3:
            data_acc['ann']=acc
            data_recall['ann']=recall
    output_acc.append(data_acc)
    output_recall.append(data_recall)  

dataf_acc=pd.DataFrame(output_acc)
dataf_recall=pd.DataFrame(output_recall)
print("******** acc ********")
print(dataf_acc)
print("******** recall ********")
print(dataf_recall)
dataf_acc.to_csv('dataf_acc.csv',index=False)
dataf_recall.to_csv('dataf_recall.csv',index=False)
