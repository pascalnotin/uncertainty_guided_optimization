import math, random, sys, os
import argparse
import numpy as np
import pandas as pd
import rdkit.Chem.QED
from rdkit.Chem import AllChem as Chem
from rdkit import RDLogger

import torch
import torch.nn as nn

from JTVAE.fast_jtnn import *
from utils import quality_filters as qual, optimization_utils as ou

def main(vocab_path, output_file, model_path, nsample, hidden_size=450, latent_size=56, depthT=20, depthG=3, dropout_rate_GRU=0.0, dropout_rate_MLP=0.0, prop=None, drop_prop_NN=0.0):
    vocab = [x.strip("\r\n ") for x in open(vocab_path)] 
    vocab = Vocab(vocab)
    output_folder=os.path.dirname(model_path)+os.sep+'samples'
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    vectorized_verify_smiles = np.vectorize(ou.verify_smile)
    vectorized_compute_target_logP = np.vectorize(ou.compute_target_logP)
    vectorized_compute_qed = np.vectorize(ou.compute_qed)
    QF = qual.QualityFiltersCheck(training_data_smi=[])

    if prop is None:
        model = JTNNVAE(vocab, hidden_size, latent_size, depthT, depthG, dropout_rate_GRU, dropout_rate_MLP)
    else:
        model = JTNNVAE_prop(vocab, hidden_size, latent_size, depthT, depthG, prop=prop, dropout_rate_GRU=dropout_rate_GRU, dropout_rate_MLP=dropout_rate_MLP, drop_prop_NN=drop_prop_NN)
    dict_buffer = torch.load(model_path)
    model.load_state_dict(dict_buffer)
    model = model.cuda()

    torch.manual_seed(0)

    list_sampled_molecules = []

    with open(output_folder+os.sep+'samples_'+str(nsample)+'_model_'+str(model_path.split(os.sep)[-2])+'_'+str(output_file)+'.txt', 'w') as out_file:
        for i in range(nsample):
            new_molecule = str(model.sample_prior())
            list_sampled_molecules.append(new_molecule)
            out_file.write(new_molecule+'\n')
        
        out_file.write('\n\n')
        
        out_file.write("Percent valid molecules: "+str(np.array(vectorized_verify_smiles(list_sampled_molecules)).sum()/len(list_sampled_molecules))+'\n')
        out_file.write("Avg penalized logP: "+str(np.mean(vectorized_compute_target_logP(list_sampled_molecules)))+'\n')
        out_file.write("Avg QED: "+str(np.mean(vectorized_compute_qed(list_sampled_molecules)))+'\n')
        out_file.write("Avg quality: "+str(np.array(QF.check_smiles_pass_quality_filters_flag(list_sampled_molecules)).sum()/len(list_sampled_molecules)))


if __name__ == '__main__':
    lg = RDLogger.logger() 
    lg.setLevel(RDLogger.CRITICAL)

    parser = argparse.ArgumentParser()
    parser.add_argument('--nsample', type=int, required=True)
    parser.add_argument('--vocab_path', required=True)
    parser.add_argument('--model_checkpoint', required=True)
    parser.add_argument('--output_file', required=True)
    
    parser.add_argument('--hidden_size', type=int, default=450)
    parser.add_argument('--latent_size', type=int, default=56)
    parser.add_argument('--depthT', type=int, default=20)
    parser.add_argument('--depthG', type=int, default=3)
    parser.add_argument('--dropout_rate_GRU', type=float, default=0.0)
    parser.add_argument('--dropout_rate_MLP', type=float, default=0.0)
    
    parser.add_argument('--property', type=str, default=None)
    parser.add_argument('--drop_prop_NN', type=float, default=0.0)

    args = parser.parse_args()
    
    main(args.vocab_path, args.output_file, args.model_checkpoint, args.nsample, args.hidden_size, args.latent_size, args.depthT, args.depthG, args.dropout_rate_GRU, args.dropout_rate_MLP, args.property, args.drop_prop_NN)