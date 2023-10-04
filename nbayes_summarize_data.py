#creates a summary of prob for each feature and class in train data
#output "values, clump, uniformity,marginal, mitoses"
#class label: 2- bening, 4-malignant, 1-10 for the others
# Author: Petter Byström
from collections import Counter
import numpy as np
import pandas as pd
import os
import sys
import argparse

features = ["clump", "uniformity", "marginal", "mitoses","class"]
values = range(1,11)

def data_from_tabtxt(file):
    data = pd.read_csv(file,delimiter='\t',header=None, names=features).fillna(0)
    return data

def split_class_labels(data):
    data_b = data[data['class']==2]
    data_m = data[data['class']==4]
    print(len(data_b),len(data_m))
    return data_b,data_m

def probability(data):
    prob = pd.DataFrame(columns=[features[:-1]])
    for val in values:
        prob.loc[val] = [f'{round(Counter(data[i] == val)[True]/(Counter(data[i] == val)[True]+Counter(data[i] == val)[False]-Counter(data[i]==0)[True]),3):.3f}' for i in features[:-1]]
    return prob

if __name__ == '__main__':

    # Set up the parsing of command-line arguments
    parser = argparse.ArgumentParser(
        description="Native bayes summarize"
    )
    parser.add_argument(
        "--traindir",
        required=True,
        help="Path to input directory containing tumor info file "
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Path to output directory "
    )

    args = parser.parse_args()
    train_dir = args.traindir
    out_dir = args.outdir
    os.makedirs(args.outdir, exist_ok=True)
    try:
        file_name1 = "{}/output_summary_class_2.txt".format(args.outdir)
        file_name2 = "{}/output_summary_class_4.txt".format(args.outdir)
        data = data_from_tabtxt('{}/tumor_info.txt'.format(args.traindir))
        split_data= split_class_labels(data)
        probability(split_data[0]).to_csv(file_name1, header=features[:-1], index=True, sep='\t', mode='w')
        probability(split_data[1]).to_csv(file_name2, header=features[:-1], index=True, sep='\t', mode='w')
    except IOError:
        print("Output file {} cannot be created".format(file_name1+file_name2))
        sys.exit(1)
    


