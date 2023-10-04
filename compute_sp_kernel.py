"""
Homework  : Similarity measures on sets
Course    : Data Mining (636-0018-00L)

Compute sp kernel


Student: Petter Byström
"""
from shortest_path_kernel import floyd_warshall
from shortest_path_kernel import sp_kernel
import os
import sys
import argparse
import numpy as np
import scipy.io 


if __name__ == '__main__':

    # Set up the parsing of command-line arguments
    parser = argparse.ArgumentParser(
        description="Compute SP kernel"
    )
    parser.add_argument(
        "--datadir",
        required=True,
        help="Path to input directory containing file MUTAG.mat"
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Path to directory where graph_output.txt will be created"
    )

    args = parser.parse_args()

    # Set the paths
    data_dir = args.datadir
    out_dir = args.outdir
    os.makedirs(args.outdir, exist_ok=True)

    mat = scipy.io.loadmat("{}/{}".format(data_dir,'MUTAG.mat'))
    label = np.reshape(mat['lmutag'],(len(mat['lmutag'], )))
    sp_matrix = np.asarray([floyd_warshall(matrix) for matrix in np.reshape(mat['MUTAG']['am'],(len(label), ) )])
    try:
        file_name = "{}/graphs_output.txt".format(args.outdir)
        f_out = open(file_name, 'w')
    except IOError:
        print("Output file {} cannot be created".format(file_name))
        sys.exit(1)
    f_out.write('{}\t{}\n'.format('Pair of classes','SP'))

    
    cdict = {}
    cdict['normal'] = -1
    cdict['abnormal'] = 1
    lst_group = ['abnormal', 'normal']

    for idx_g1 in range(len(lst_group)):
        for idx_g2 in range(idx_g1, len(lst_group)):
            group1 = sp_matrix[label == cdict[lst_group[idx_g1]]]
            group2 = sp_matrix[label == cdict[lst_group[idx_g2]]]
            count = 0
            vec = np.zeros(1,dtype=float)
            for x in group1:
                for y in group2:
                    if idx_g1 == idx_g2 and np.array_equal(x,y):
                        continue
                    vec[0] += sp_kernel(x,y)
                    count += 1
            vec /= count
            str = '\t'.join('{0:.2f}'.format(x) for x in vec)
            f_out.write(
                '{}:{}\t{}\n'.format(
                    lst_group[idx_g1], lst_group[idx_g2], str))
    f_out.close()
