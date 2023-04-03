import pandas as pd
from multiprocessing import Pool

def process_data(params):
    i, k, res = params
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    for j in range(0, 33748):
        if res.loc[j][k]==i and res.loc[j][4]==i:
            tp = tp + 1
        if res.loc[j][k]==i and res.loc[j][4]!=i:
            fn = fn + 1
        if res.loc[j][k]!=i and res.loc[j][4]==i:
            fp = fp + 1
        if res.loc[j][k]!=i and res.loc[j][4]!=i:
            tn = tn + 1
    acc = (tp + tn) / (tp + tn + fp + fn)
    recall = tp / (tp + fn)
    return (i, k, acc, recall)

if __name__ == '__main__':
    res = pd.read_csv('checkclass.csv')
    params_list = [(i, k, res) for i in range(26) for k in range(1, 4)]
    
    with Pool(processes=50) as pool:
        result = pool.map(process_data, params_list)
        

    data_acc = {}
    data_recall = {}
    for i, k, acc, recall in result:
        if k == 1:
            data_acc[i] = {'xgb': acc}
            data_recall[i] = {'xgb': recall}
        elif k == 2:
            data_acc[i]['nn'] = acc
            data_recall[i]['nn'] = recall
        else:
            data_acc[i]['ann'] = acc
            data_recall[i]['ann'] = recall

    dataf_acc=pd.DataFrame(data_acc).T
    dataf_recall=pd.DataFrame(data_recall).T
    print("******** acc ********")
    print(dataf_acc)
    print("******** recall ********")
    print(dataf_recall)
    dataf_acc.to_csv('dataf_acc.csv',index=True)
    dataf_recall.to_csv('dataf_recall.csv',index=True)
