import math, random, sys, os
import numpy as np
import argparse
from collections import deque
import pickle as pickle
import rdkit
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.autograd import Variable

from JTVAE.fast_jtnn import *

if __name__ == '__main__':
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', required=True)
    parser.add_argument('--vocab_path', required=True)
    parser.add_argument('--save_path', required=True)
    parser.add_argument('--load_epoch', type=int, default=0)

    parser.add_argument('--hidden_size', type=int, default=450)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--latent_size', type=int, default=56)
    parser.add_argument('--depthT', type=int, default=20)
    parser.add_argument('--depthG', type=int, default=3)
    parser.add_argument('--dropout_rate_GRU', type=float, default=0.0)
    parser.add_argument('--dropout_rate_MLP', type=float, default=0.0)

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--clip_norm', type=float, default=50.0)
    parser.add_argument('--beta', type=float, default=0.0)
    parser.add_argument('--step_beta', type=float, default=0.002)
    parser.add_argument('--max_beta', type=float, default=1.0)
    parser.add_argument('--warmup', type=int, default=40000)

    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--anneal_rate', type=float, default=0.9)
    parser.add_argument('--anneal_iter', type=int, default=40000)
    parser.add_argument('--kl_anneal_iter', type=int, default=2000)
    parser.add_argument('--print_iter', type=int, default=50)

    args = parser.parse_args()
    print(args)
        
    vocab = [x.strip("\r\n ") for x in open(args.vocab_path)] 
    vocab = Vocab(vocab)

    model = JTNNVAE(vocab, int(args.hidden_size), int(args.latent_size), int(args.depthT), int(args.depthG), dropout_rate_GRU=args.dropout_rate_GRU, dropout_rate_MLP=args.dropout_rate_MLP).cuda()
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    model.save_params(save_path=args.save_path, name_parameter_file='parameters.json')
    
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)
    
    if args.load_epoch > 0:
        model.load_state_dict(torch.load(args.save_path + "/model.epoch-" + str(args.load_epoch)))

    print("Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,))

    model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)

    param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
    grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

    total_step = args.load_epoch
    beta = args.beta
    meters = np.zeros(4)
    
    loader = MolTreeFolder(args.train_path, vocab, args.batch_size, num_workers=14)

    for epoch in tqdm(range(args.epoch)): 
        print("Starting epoch: "+str(epoch))   
        for batch in loader:
            total_step += 1
            try:
                model.zero_grad()
                loss, kl_div, wacc, tacc, sacc = model(batch, beta)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
                optimizer.step()
            except Exception as e:
                print(e)
                continue

            meters = meters + np.array([kl_div, wacc * 100, tacc * 100, sacc * 100])

            if total_step % args.print_iter == 0:
                meters /= args.print_iter
                print("[%d] Beta: %.3f, KL: %.2f, Word: %.2f, Topo: %.2f, Assm: %.2f, PNorm: %.2f, GNorm: %.2f" % (total_step, beta, meters[0], meters[1], meters[2], meters[3], param_norm(model), grad_norm(model)))
                sys.stdout.flush()
                meters *= 0

            if total_step % args.anneal_iter == 0:
                scheduler.step()
                print("learning rate: %.6f" % scheduler.get_last_lr()[0])

            if total_step % args.kl_anneal_iter == 0 and total_step >= args.warmup:
                beta = min(args.max_beta, beta + args.step_beta)
        torch.save(model.state_dict(), args.save_path + "/model.epoch-" + str(epoch))
    
