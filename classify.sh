###module 1
cp  ${fasta_file}  yes.fa
####CNN

 python seq2pad.py  yes.fa  fa.pt
 python two_cnn.py
rm -f fa.pt
echo " ROS-CNN finish(1/2)"
###NN
python ~/software/iFeature-master/iFeature.py  --file  yes.fa --type DPC --out 01
python scale.py
python nn_test.py  > nn_2class.out
rm -f 01 DPC.out
echo " ROS-NN finish(1/2)"
##XGBOOST
python ~/software/iFeature-master/iFeature.py  --file  yes.fa  --type CKSAAGP  --out CKSAAGP.out
python 2classxgb.py
rm -f CKSAAGP.out
echo " ROS-XGBOOST (1/2)"
###
# hard voting
paste xgb_2class.out nn_2class.out cnn_2class.out |awk '{print $1" "$2+$3+$4}'|awk '$2>1'  |awk '{print $1}' >yes.id
seqkit grep -f yes.id yes.fa  > yes-yes.fa
rm -f yes.id
echo " hard voting finish"
###########################################################module 2
####CNN
python seq2pad.py yes-yes.fa yes.fa.pt
python N_cnn.py 
rm -f yes.fa.pt
echo " ROS-CNN finish(2/2)"
####XGboost iFeature.py
python ~/software/iFeature-master/iFeature.py  --file   yes-yes.fa --type CKSAAGP  --out CKSAAGP.out
python Nclassxgb.py
rm -f CKSAAGP.out
echo "  ROS-XGBOOST (2/2)"

####NN
python  iFeature.py  --file   yes-yes.fa --type DPC --out 01
python scale.py
python N_nn_test.py
echo "ROS-NN finish(2/2)"
#### soft voting
rm -f yes-yes.fa
python soft_vote.py
echo " soft voting finish"
rm -f blastros.out.list yes.fa nn_2class.out 01 DPC.out N_nn.out N_nn.res xgb_Nclass.res xgb_Nclass.out N_cnn.out cnn_2class.proba cnn_2class.out  xgb_2class.out
#####
cat final_Nclass.out
