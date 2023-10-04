# Script for K-nn algorithm
#Author: Petter Byström
from collections import Counter
import numpy as np
import pandas as pd
import os
import sys
import argparse

def data_from_tabtxt(file):
    data = pd.read_csv(file,delimiter='\t') 
    return data

def evaluate_classification(true, pred):
    positive_indexes = true['ERstatus'] == '+' 
    negative_indexes = true['ERstatus']== '-'
    tp = sum(pred[positive_indexes] == '+') 
    tn = sum(pred[negative_indexes] == '-') 
    fp = sum(pred[negative_indexes] == '+') 
    fn = sum(pred[positive_indexes] == '-') 
    return [tp,tn],[fp,fn]

def accuracy(true_p_n,false_p_n):
    return sum(true_p_n)/(sum(true_p_n)+sum(false_p_n))
def precision(true_p_n,false_p_n):
    return true_p_n[0]/(true_p_n[0]+false_p_n[0])
def recall(true_p_n,false_p_n):
    return true_p_n[0]/(true_p_n[0]+false_p_n[1])

def classify(neigh):
    count = Counter(neigh)
    if len(neigh)%2 == 0 and count.most_common(1)[0][1] == len(neigh)/2 :
        count = Counter(neigh[:-1])
    return count.most_common(1)[0][0]

def eucledian(test_row,train_x,k):
    dist = [(i,np.sqrt(np.sum(np.square(test_row-train_row)))) for i,train_row in train_x.drop(columns=['patientId']).iterrows()]
    dist.sort(key=lambda tup: tup[1])
    return [dist[i] for i in range(k)]

def knn(train_x,train_y,test_x,k):
    nearest_neigh=[]
    for index,row in test_x.drop(columns=['patientId']).iterrows():
        nearest_neigh.append(eucledian(row,train_x,k))

    prediction_df = pd.DataFrame(
        ([classify([train_y.loc[inx].at["ERstatus"] for inx, dist in i]) for i in nearest_neigh ]),
        columns=['prediction']
    )
    return prediction_df



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Compute k-NN"
    )
    parser.add_argument(
        "--traindir",
        required=True,
        help="Path to input directory containing train file "
    )
    
    parser.add_argument(
        "--testdir",
        required=True,
        help="Path to input directory containing test file "
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Path to directory where output_knn.txt will be created"
    )
    parser.add_argument(
        "--mink",
        required=True,
        help="minimum value of k"
    )
    parser.add_argument(
        "--maxk",
        required=True,
        help="maximum value of k"
    )
    args = parser.parse_args()
    train_dir = args.traindir
    test_dir = args.testdir
    out_dir = args.outdir
    min_k = int(args.mink)
    max_k = int(args.maxk)
    os.makedirs(args.outdir, exist_ok=True)
    try:
        file_name = "{}/output_knn.txt".format(args.outdir)
        f_out = open(file_name, 'w')
    except IOError:
        print("Output file {} cannot be created".format(file_name))
        sys.exit(1)
    f_out.write('{}\t{}\t{}\t{}\n'.format('Value of k','Accuracy','Precision','Recall'))

    # load data and label
    train_data = data_from_tabtxt("{}/matrix_mirna_input.txt".format(train_dir))
    test_data =data_from_tabtxt("{}/matrix_mirna_input.txt".format(test_dir))
    train_label = data_from_tabtxt("{}/phenotype.txt".format(train_dir))
    test_label = data_from_tabtxt("{}/phenotype.txt".format(test_dir))
    output_list=[]
    for i in range(min_k,max_k+1):
        pred = knn(train_data,train_label,test_data,i)
        true_p_n,false_p_n = evaluate_classification(test_label,pred['prediction'])
        output_list.append([i,accuracy(true_p_n,false_p_n),precision(true_p_n,false_p_n),recall(true_p_n,false_p_n)])
        true_p_n,false_p_n=[0,0],[0,0]
    for i in output_list:
        f_out.write('{}\t{}\t{}\t{}\n'.format(round(i[0],2),round(i[1],2),round(i[2],2),round(i[3],2)))
    f_out.close()