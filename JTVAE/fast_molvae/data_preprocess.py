import sys,os
import math,random
import pickle
import numpy as np
import rdkit
from multiprocessing import Pool
from optparse import OptionParser
from tqdm import tqdm

import torch
import torch.nn as nn

from JTVAE.fast_jtnn import *

def tensorize(smiles, assm=True):
    mol_tree = MolTree(smiles)
    mol_tree.recover()
    if assm:
        mol_tree.assemble()
        for node in mol_tree.nodes:
            if node.label not in node.cands:
                node.cands.append(node.label)

    del mol_tree.mol
    for node in mol_tree.nodes:
        del node.mol
    return mol_tree

if __name__ == "__main__":
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)
    
    parser = OptionParser()
    parser.add_option("-t", "--train", dest="train_path")
    parser.add_option("-n", "--split", dest="nsplits", default=10)
    parser.add_option("-j", "--jobs", dest="njobs", default=8)
    parser.add_option("-o", "--output", dest="output_path")
    opts, args = parser.parse_args()
    opts.njobs = int(opts.njobs)

    pool = Pool(opts.njobs)
    num_splits = int(opts.nsplits)
    
    out_path = os.path.join(opts.output_path, './')
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    
    with open(opts.train_path) as f:
        data = [line.strip("\r\n ").split()[0] for line in f]
    
    all_data = pool.map(tensorize, data)
    num_errors = np.array([x for x in all_data if x is None]).sum()
    print("Number of input molecules that could not be pre-processed: "+str(num_errors))
    all_data = [x for x in all_data if x is not None]
    all_data_split = np.array_split(all_data, num_splits)
    
    for split_id in tqdm(range(num_splits)):
        with open(os.path.join(opts.output_path, 'tensors-%d.pkl' % split_id), 'wb') as f:
            pickle.dump(all_data_split[split_id], f)