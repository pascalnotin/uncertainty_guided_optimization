import math, random, sys, os
from optparse import OptionParser
from collections import deque
import argparse
import tqdm
from rdkit import RDLogger

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.autograd import Variable

from JTVAE.fast_jtnn import *

if __name__ == '__main__':
    lg = RDLogger.logger() 
    lg.setLevel(RDLogger.CRITICAL)

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--vocab_path", required=True)
    parser.add_argument("--prop_path", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=450)
    parser.add_argument("--latent_size", type=int, default=56)
    parser.add_argument("--depthT", type=int, default=20)
    parser.add_argument("--depthG", type=int, default=3)
    parser.add_argument('--property', type=str, default="penalized_logP")
    parser.add_argument('--dropout_rate_GRU', type=float, default=0.0)
    parser.add_argument('--dropout_rate_MLP', type=float, default=0.0)
    parser.add_argument('--drop_prop_NN', type=float, default=0.0)
    parser.add_argument('--pretrained_model_path', default=None, help="Path to pre-trained model (eg., pre-trained model without KL term)")
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    
    vocab = [x.strip("\r\n ") for x in open(args.vocab_path)] 
    vocab = Vocab(vocab)

    batch_size = int(args.batch_size)
    hidden_size = int(args.hidden_size)
    latent_size = int(args.latent_size)
    depthT = int(args.depthT)
    depthG = int(args.depthG)
    beta = float(args.beta)
    lr = float(args.lr)
    
    model = JTNNVAE_prop(vocab, hidden_size, latent_size, depthT, depthG, prop=args.property, dropout_rate_GRU=args.dropout_rate_GRU, dropout_rate_MLP=args.dropout_rate_MLP, drop_prop_NN=args.drop_prop_NN)
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    model.save_params(save_path=args.save_path, name_parameter_file='parameters.json')

    if args.pretrained_model_path is not None:
        model.load_state_dict(torch.load(args.pretrained_model_path))
        print("Loading model from checkpoint")
    else:
        for param in model.parameters():
            if param.dim() == 1:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_normal_(param)

    model = model.cuda()
    print("Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,))

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, 0.9)

    dataset = PropDataset(args.train_path, args.prop_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=lambda x:x, drop_last=True)

    MAX_EPOCH = 17
    PRINT_ITER = 50

    for epoch in tqdm.tqdm(range(MAX_EPOCH)):
        print("Starting epoch: "+str(epoch))
        word_acc,topo_acc,assm_acc,prop_acc = 0,0,0,0

        for it, batch in enumerate(dataloader):
            for mol_tree,_ in batch:
                for node in mol_tree.nodes:
                    if node.label not in node.cands:
                        node.cands.append(node.label)

            model.zero_grad()
            loss, kl_div, wacc, tacc, sacc, pacc = model(batch, beta)
            loss.sum().backward()
            optimizer.step()

            word_acc += wacc
            topo_acc += tacc
            assm_acc += sacc
            prop_acc += pacc

            if (it + 1) % PRINT_ITER == 0:
                word_acc = word_acc / PRINT_ITER * 100
                topo_acc = topo_acc / PRINT_ITER * 100
                assm_acc = assm_acc / PRINT_ITER * 100
                prop_acc /= PRINT_ITER

                iter_step = it + 1
                print("Iter: %d, KL: %.1f, Word: %.2f, Topo: %.2f, Assm: %.2f, Prop: %.4f" % (iter_step, kl_div, word_acc, topo_acc, assm_acc, prop_acc))
                word_acc,topo_acc,assm_acc,prop_acc = 0,0,0,0
                sys.stdout.flush()

            if (it + 1) % 1500 == 0: #Fast annealing
                scheduler.step()
                print("learning rate: %.6f" % scheduler.get_last_lr()[0])

        scheduler.step()
        print("learning rate: %.6f" % scheduler.get_last_lr()[0])
        torch.save(model.state_dict(), args.save_path + "/model.epoch-" + str(epoch))
    torch.save(model.state_dict(), args.save_path + "/model.final")