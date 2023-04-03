import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
N_nn_out = pd.read_csv("N_nn.out", sep=' ', header=0)
scaler = MinMaxScaler(feature_range=(0, 1))
N_nn_out=scaler.fit_transform(N_nn_out)
N_nn_out=pd.DataFrame(N_nn_out)
N_cnn_out = pd.read_csv("N_cnn.out", sep=' ', header=0)
xgb_Nclass_out = pd.read_csv("xgb_Nclass.out", sep=' ', header=0)
id=pd.DataFrame(xgb_Nclass_out['#'])

xgb_Nclass_out = xgb_Nclass_out.drop(columns=['#'])
new_df = N_cnn_out + xgb_Nclass_out
#new_df=new_df+N_nn_out
N_nn_out = pd.DataFrame(np.random.random(new_df.shape),columns=new_df.columns)
new_df=new_df+N_nn_out
rosclass=pd.DataFrame(new_df.idxmax(1))
#id=pd.DataFrame(xgb_Nclass_out['#'])
f1=pd.concat([id, rosclass], axis=1)
#f1.rename(columns={'#':'seq_id', '0':'class'}, inplace=True) 
f1_col=["seq_id","class"]
f1.columns=f1_col
#f1.replace({"0":"thioredoxin reductase","1":"cytochrome c peroxidase"}
f1.replace({"0":"thioredoxin reductase", "1":"cytochrome c peroxidase", "2":"peroxidase", "3":"glutathione peroxidase", "4":"nickel superoxide dismutase", "5":"alkyl hydroperoxide reductase", "6":"thioredoxin 1", "7":"thioredoxin 2", "8":"glutaredoxin 1", "9":"glutaredoxin 2", "10":"catalase", "11":"catalase-peroxidase", "12":"superoxide dismutase 2", "13":"superoxide dismutase 1", "14":"NADH peroxidase", "15":"superoxide reductase", "16":"Mn-containing catalase", "17":"monothiol glutaredoxin", "18":"thiol peroxidase", "19":"peroxiredoxin 5", "20":"peroxiredoxin 6", "21":"peroxiredoxin 1", "22":"alkyl hydroperoxide reductase 1", "23":"rubrerythrin", "24":"peroxiredoxin 3", "25":"glutaredoxin 3"}

        
        
        
        ,inplace=True)




f1.to_csv("final_Nclass.out",sep="\t",index=False)
