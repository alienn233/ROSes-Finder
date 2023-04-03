import pandas as pd
import xgboost as xgb

import numpy as np
import joblib
from xgboost import XGBClassifier
res=[]
clf = joblib.load('xgboos_Nclass.pkl')
data1 = pd.read_csv("CKSAAGP.out", sep='\t', header=0)
X = data1.drop(columns=['#'])
dtest = xgb.DMatrix(X)

y_pred=clf.predict(dtest)
#AAA=[round(value) for value in y_pred]
#info=pd.value_counts(AAA)#1 0 number
#a1=pd.DataFrame(AAA)
#a2=pd.DataFrame(data1["#"])
#a3=pd.concat([a2, a1], axis=1)
f2=pd.DataFrame(y_pred)
X=data1['#']
x=pd.DataFrame(X)
a3=pd.concat([x, f2], axis=1)
a3.to_csv("xgb_Nclass.out",sep=" ",index=False)
f3=pd.DataFrame(f2.idxmax(1))
a4=pd.concat([x, f3], axis=1)
a4.to_csv("xgb_Nclass.res",sep="\t",header=None,index=False)


