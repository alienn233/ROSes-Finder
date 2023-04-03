import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
data = pd.read_csv("01",sep="\t")
data=data.drop(columns=['#'])
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)
data=pd.DataFrame(data)
data=pd.DataFrame(data)
data.to_csv("DPC.out",sep="\t")



