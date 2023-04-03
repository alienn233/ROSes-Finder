# ROSes-Finder
The model mainly consists of two modules for classifying FASTA files. The whole process includes sequence processing, feature extraction, training and prediction using different models based on natural language processing, and selecting the classification results of different machine learning models by voting.

# Outline for README

- Requirements
- Run Scripts
- Workflow
	- -model1
	- -model2
- Output

## Requirements

- Python  3.7.11
- XGBoost 1.6.1
- NumPy 1.17.3
- scikit-learn 0.24.2
- iFeature


##  Run Scripts

```
bash classify.sh <fasta_file> 
```
- fasta_file：The path to the input fasta file.
- example: bash  classify.sh test.fa

## Workflow
The script consists of two main modules, each of which contains three classifiers.

### model1
- CNN
- ANN
- XGBoost

In this module, natural language processing techniques are used to extract Dipeptide Composition (DPC) features from fasta files using iFeature. The features are then normalized and used as inputs for the Neural Network (NN) classifier. The Composition, Transition, and Distribution (CTD) features are also normalized and used as inputs for the XGBoost classifier. The results are then subjected to hard voting, and fasta sequences identified as ROSes are stored.
### model2
- CNN
- ANN
- XGBoost


This module takes the selected sequences from Module 1 as input and performs data processing and feature extraction in a similar manner as Module 1. The processed data is subjected to multiclass classification. Finally, the results of the three classifiers are subjected to soft voting to obtain the final result.

## Output
The final classification results are stored in 'final_Nclass.out'.
## ROS-CNN model download address
Due to GitHub's file size limitations, the ROSes-CNN model can be downloaded via Google Drive
https://drive.google.com/file/d/1FDO7oSvnasWIvWRsaHGw1Nnc-KsVK7b-/view?usp=share_link
https://drive.google.com/file/d/1rNKbvN9k0R2m0OoDnRT1oaX-KO_q4aS-/view?usp=share_link
