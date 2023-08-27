
f1.replace({"0":"thioredoxin reductase", "1":"cytochrome c peroxidase", "2":"peroxidase", "3":"glutathione peroxidase", "4":"nickel superoxide dismutase", "5":"alkyl hydroperoxide reductase", "6":"thioredoxin 1", "7":"thioredoxin 2", "8":"glutaredoxin 1", "9":"glutaredoxin 2", "10":"catalase", "11":"catalase-peroxidase", "12":"superoxide dismutase 2", "13":"superoxide dismutase 1", "14":"NADH peroxidase", "15":"superoxide reductase", "16":"Mn-containing catalase", "17":"monothiol glutaredoxin", "18":"thiol peroxidase", "19":"peroxiredoxin 5", "20":"peroxiredoxin 6", "21":"peroxiredoxin 1", "22":"alkyl hydroperoxide reductase 1", "23":"rubrerythrin", "24":"peroxiredoxin 3", "25":"glutaredoxin 3"}

        
        
        
        ,inplace=True)




f1.to_csv("final_Nclass.out",sep="\t",index=False)
