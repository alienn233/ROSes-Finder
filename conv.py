import os
import sys

fasta_file = sys.argv[1]


# CNN
#os.system("python seq2pad.py test.fa fa.pt")
#os.system("python two_cnn.py")
#os.system("rm -f fa.pt")

# NN
os.system("python ~/software/iFeature-master/iFeature.py --file {} --type DPC --out 01".format(fasta_file))
os.system("python scale.py")
os.system("python nn_test.py > nn_2class.out")
os.system("rm -f 01 DPC.out")

# XGBOOST
os.system("python ~/software/iFeature-master/iFeature.py --file {} --type CKSAAGP --out CKSAAGP.out".format(fasta_file))
os.system("python 2classxgb.py")
os.system("rm -f CKSAAGP.out")

# Hard voting
os.system("paste xgb_2class.out nn_2class.out cnn_2class.out |awk '{print $1\" \"$2+$3+$3}'|awk '$2>1'|awk '{print $1}' >yes.id")
os.system("perl extrac.pl yes.id test.fa >yes.fa")

# Module 2
#CNN
#os.system("python seq2pad.py yes.fa yes.fa.pt")
#os.system("python N_cnn.py")
#os.system("rm -f yes.fa.pt")

# XGboost
os.system("python ~/software/iFeature-master/iFeature.py --file yes.fa --type CKSAAGP --out CKSAAGP.out")
os.system("python Nclassxgb.py")
os.system("rm -f CKSAAGP.out")

# NN
os.system("python ~/software/iFeature-master/iFeature.py --file yes.fa --type DPC --out 01")
os.system("python scale.py")
os.system("python N_nn_test.py")

# Soft voting
os.system("python soft_vote.py")
os.system("rm -f 01 CKSAAGP.out xgb_Nclass.out DPC.out N_nn.res N_nn.out nn_2class.out")
