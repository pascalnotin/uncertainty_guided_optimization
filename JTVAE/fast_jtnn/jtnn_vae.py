import os
import json
import numpy as np
import tqdm
import copy, math
import rdkit
import rdkit.Chem as Chem
from rdkit.Chem import Descriptors

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions.multivariate_normal as rnv

from JTVAE.fast_jtnn.mol_tree import Vocab, MolTree
from JTVAE.fast_jtnn.nnutils import create_var, flatten_tensor, avg_pool
from JTVAE.fast_jtnn.jtnn_enc import JTNNEncoder
from JTVAE.fast_jtnn.jtnn_dec import JTNNDecoder
from JTVAE.fast_jtnn.mpn import MPN
from JTVAE.fast_jtnn.jtmpn import JTMPN
from JTVAE.fast_jtnn.datautils import tensorize, smiles_to_moltree
from JTVAE.fast_jtnn.chemutils import enum_assemble, set_atommap, copy_edit_mol, attach_mols

from utils import optimization_utils as ou

class JTNNVAE(nn.Module):

    def __init__(self, vocab, hidden_size, latent_size, depthT, depthG, dropout_rate_GRU=0.0, dropout_rate_MLP=0.0):
        super(JTNNVAE, self).__init__()
        self.vocab = vocab
        self.hidden_size = hidden_size
        self.latent_size = latent_size = latent_size // 2
        self.depthT=depthT
        self.depthG=depthG
        self.dropout_rate_GRU=dropout_rate_GRU
        self.dropout_rate_MLP=dropout_rate_MLP

        self.jtnn = JTNNEncoder(hidden_size, depthT, nn.Embedding(vocab.size(), hidden_size))   
        self.decoder = JTNNDecoder(vocab, hidden_size, latent_size, nn.Embedding(vocab.size(), hidden_size), dropout_rate_GRU=dropout_rate_GRU, dropout_rate_MLP=dropout_rate_MLP)
        
        self.jtmpn = JTMPN(hidden_size, depthG) 
        self.mpn = MPN(hidden_size, depthG)         

        self.A_assm = nn.Linear(latent_size, hidden_size, bias=False)
        self.assm_loss = nn.CrossEntropyLoss(reduction='sum')

        self.T_mean = nn.Linear(hidden_size, latent_size)
        self.T_var = nn.Linear(hidden_size, latent_size)
        self.G_mean = nn.Linear(hidden_size, latent_size)
        self.G_var = nn.Linear(hidden_size, latent_size)
    
    def save_params(self, save_path, name_parameter_file):
        params = {
            'hidden_size': self.hidden_size, 
            'latent_size': int(self.latent_size * 2),
            'depthT': self.depthT, 
            'depthG': self.depthG, 
            'dropout_rate_GRU': self.dropout_rate_GRU,
            'dropout_rate_MLP': self.dropout_rate_MLP
        }
        with open(save_path + os.sep + name_parameter_file, 'w') as fp:
            json.dump(params, fp)

    def encode(self, jtenc_holder, mpn_holder):
        tree_vecs, tree_mess = self.jtnn(*jtenc_holder)
        mol_vecs = self.mpn(*mpn_holder)
        return tree_vecs, tree_mess, mol_vecs
    
    def encode_from_smiles(self, smiles_list):
        tree_batch = [MolTree(s) for s in smiles_list]
        _, jtenc_holder, mpn_holder = tensorize(tree_batch, self.vocab, assm=False)
        tree_vecs, _, mol_vecs = self.encode(jtenc_holder, mpn_holder)
        return torch.cat([tree_vecs, mol_vecs], dim=-1)

    def encode_latent(self, jtenc_holder, mpn_holder):
        tree_vecs, _ = self.jtnn(*jtenc_holder)
        mol_vecs = self.mpn(*mpn_holder)
        tree_mean = self.T_mean(tree_vecs)
        mol_mean = self.G_mean(mol_vecs)
        tree_var = -torch.abs(self.T_var(tree_vecs))
        mol_var = -torch.abs(self.G_var(mol_vecs))
        return torch.cat([tree_mean, mol_mean], dim=1), torch.cat([tree_var, mol_var], dim=1)

    def rsample(self, z_vecs, W_mean, W_var):
        batch_size = z_vecs.size(0)
        z_mean = W_mean(z_vecs)
        z_log_var = -torch.abs(W_var(z_vecs)) #Following Mueller et al.
        kl_loss = -0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)) / batch_size
        epsilon = create_var(torch.randn_like(z_mean))
        z_vecs = z_mean + torch.exp(z_log_var / 2) * epsilon
        return z_vecs, kl_loss
    
    def encode_and_samples_from_smiles(self, smiles_list):
        tree_batch = [MolTree(s) for s in smiles_list]
        _, jtenc_holder, mpn_holder = tensorize(tree_batch, self.vocab, assm=False)
        tree_vecs, _, mol_vecs = self.encode(jtenc_holder, mpn_holder)
        z_tree_vecs,_ = self.rsample(tree_vecs, self.T_mean, self.T_var)
        z_mol_vecs,_ = self.rsample(mol_vecs, self.G_mean, self.G_var)
        return torch.cat([z_tree_vecs, z_mol_vecs], dim=1)

    def sample_prior(self, prob_decode=False):
        z_tree = torch.randn(1, self.latent_size).cuda()
        z_mol = torch.randn(1, self.latent_size).cuda()
        return self.decode(z_tree, z_mol, prob_decode)

    def forward(self, x_batch, beta):                                    
        x_batch, x_jtenc_holder, x_mpn_holder, x_jtmpn_holder = x_batch
        x_tree_vecs, x_tree_mess, x_mol_vecs = self.encode(x_jtenc_holder, x_mpn_holder)
        z_tree_vecs,tree_kl = self.rsample(x_tree_vecs, self.T_mean, self.T_var)
        z_mol_vecs,mol_kl = self.rsample(x_mol_vecs, self.G_mean, self.G_var)

        kl_div = tree_kl + mol_kl
        word_loss, topo_loss, word_acc, topo_acc = self.decoder(x_batch, z_tree_vecs)
        assm_loss, assm_acc = self.assm(x_batch, x_jtmpn_holder, z_mol_vecs, x_tree_mess)

        return word_loss + topo_loss + assm_loss + beta * kl_div, kl_div.item(), word_acc, topo_acc, assm_acc

    def assm(self, tree_batch, jtmpn_holder, x_mol_vecs, x_tree_mess, mode="avg_loss"):
        jtmpn_holder,batch_idx = jtmpn_holder
        fatoms,fbonds,agraph,bgraph,scope = jtmpn_holder
        batch_idx = create_var(batch_idx)

        cand_vecs = self.jtmpn(fatoms, fbonds, agraph, bgraph, scope, x_tree_mess)

        x_mol_vecs = x_mol_vecs.index_select(0, batch_idx)
        x_mol_vecs = self.A_assm(x_mol_vecs)
        scores = torch.bmm(
                x_mol_vecs.unsqueeze(1),
                cand_vecs.unsqueeze(-1)
        ).squeeze()
        
        cnt,tot,acc = 0,0,0
        if mode == "avg_loss":
            all_loss = []
        elif mode == "indep_loss":
            tree_losses = torch.zeros(len(tree_batch)).cuda()
        
        for i,mol_tree in enumerate(tree_batch):
            comp_nodes = [node for node in mol_tree.nodes if len(node.cands) > 1 and not node.is_leaf]
            cnt += len(comp_nodes)
            for node in comp_nodes:
                label = node.cands.index(node.label)
                ncand = len(node.cands)
                cur_score = scores.narrow(0, tot, ncand)
                tot += ncand

                if cur_score.data[label] >= cur_score.max().item():
                    acc += 1

                label = create_var(torch.LongTensor([label]))
                node_loss = self.assm_loss(cur_score.view(1,-1), label)
                if mode == "avg_loss":
                    all_loss.append(node_loss)
                elif mode == "indep_loss":
                    tree_losses[i] += node_loss.item()
        
        if mode == "avg_loss":
            all_loss = sum(all_loss) / len(tree_batch)
            return all_loss, acc * 1.0 / cnt
        elif mode == "indep_loss":
            return tree_losses
        

    def decode(self, x_tree_vecs, x_mol_vecs, prob_decode, fast_uncertainty_decode=False):
        assert x_tree_vecs.size(0) == 1 and x_mol_vecs.size(0) == 1

        pred_root,pred_nodes = self.decoder.decode(x_tree_vecs, prob_decode)
        if len(pred_nodes) == 0: return None
        elif len(pred_nodes) == 1: return pred_root.smiles

        for i,node in enumerate(pred_nodes):
            node.nid = i + 1
            node.is_leaf = (len(node.neighbors) == 1)
            if len(node.neighbors) > 1:
                set_atommap(node.mol, node.nid)

        scope = [(0, len(pred_nodes))]
        jtenc_holder,mess_dict = JTNNEncoder.tensorize_nodes(pred_nodes, scope)
        _,tree_mess = self.jtnn(*jtenc_holder)
        tree_mess = (tree_mess, mess_dict)

        x_mol_vecs = self.A_assm(x_mol_vecs).squeeze()

        cur_mol = copy_edit_mol(pred_root.mol)
        global_amap = [{}] + [{} for node in pred_nodes]
        global_amap[1] = {atom.GetIdx():atom.GetIdx() for atom in cur_mol.GetAtoms()}

        cur_mol,_ = self.dfs_assemble(tree_mess, x_mol_vecs, pred_nodes, cur_mol, global_amap, [], pred_root, None, prob_decode, check_aroma=True, fast_uncertainty_decode=fast_uncertainty_decode)
        if cur_mol is None: 
            cur_mol = copy_edit_mol(pred_root.mol)
            global_amap = [{}] + [{} for node in pred_nodes]
            global_amap[1] = {atom.GetIdx():atom.GetIdx() for atom in cur_mol.GetAtoms()}
            cur_mol,pre_mol = self.dfs_assemble(tree_mess, x_mol_vecs, pred_nodes, cur_mol, global_amap, [], pred_root, None, prob_decode, check_aroma=False, fast_uncertainty_decode=fast_uncertainty_decode)
            if cur_mol is None: cur_mol = pre_mol

        if cur_mol is None: 
            return None

        cur_mol = cur_mol.GetMol()
        set_atommap(cur_mol)
        cur_mol = Chem.MolFromSmiles(Chem.MolToSmiles(cur_mol))
        return Chem.MolToSmiles(cur_mol) if cur_mol is not None else None
    
    def dfs_assemble(self, y_tree_mess, x_mol_vecs, all_nodes, cur_mol, global_amap, fa_amap, cur_node, fa_node, prob_decode, check_aroma, fast_uncertainty_decode=False):
        fa_nid = fa_node.nid if fa_node is not None else -1
        prev_nodes = [fa_node] if fa_node is not None else []

        children = [nei for nei in cur_node.neighbors if nei.nid != fa_nid]
        neighbors = [nei for nei in children if nei.mol.GetNumAtoms() > 1]
        neighbors = sorted(neighbors, key=lambda x:x.mol.GetNumAtoms(), reverse=True)
        singletons = [nei for nei in children if nei.mol.GetNumAtoms() == 1]
        neighbors = singletons + neighbors

        cur_amap = [(fa_nid,a2,a1) for nid,a1,a2 in fa_amap if nid == cur_node.nid]
        cands,aroma_score = enum_assemble(cur_node, neighbors, prev_nodes, cur_amap)
        if len(cands) == 0 or (sum(aroma_score) < 0 and check_aroma):
            return None, cur_mol

        cand_smiles,cand_amap = list(zip(*cands))
        aroma_score = torch.Tensor(aroma_score).cuda()
        cands = [(smiles, all_nodes, cur_node) for smiles in cand_smiles]

        if len(cands) > 1:
            jtmpn_holder = JTMPN.tensorize(cands, y_tree_mess[1])
            fatoms,fbonds,agraph,bgraph,scope = jtmpn_holder
            cand_vecs = self.jtmpn(fatoms, fbonds, agraph, bgraph, scope, y_tree_mess[0])
            scores = torch.mv(cand_vecs, x_mol_vecs) + aroma_score
        else:
            scores = torch.Tensor([1.0])

        if prob_decode:
            probs = F.softmax(scores.view(1,-1), dim=1).squeeze() + 1e-7
            cand_idx = torch.multinomial(probs, probs.numel())
        else:
            _,cand_idx = torch.sort(scores, descending=True)

        backup_mol = Chem.RWMol(cur_mol)
        pre_mol = cur_mol

        cand_iter = min(cand_idx.numel(),1) if fast_uncertainty_decode else cand_idx.numel()

        for i in range(cand_iter):
            cur_mol = Chem.RWMol(backup_mol)
            pred_amap = cand_amap[cand_idx[i].item()]
            new_global_amap = copy.deepcopy(global_amap)

            for nei_id,ctr_atom,nei_atom in pred_amap:
                if nei_id == fa_nid:
                    continue
                new_global_amap[nei_id][nei_atom] = new_global_amap[cur_node.nid][ctr_atom]

            cur_mol = attach_mols(cur_mol, children, [], new_global_amap)
            new_mol = cur_mol.GetMol()
            new_mol = Chem.MolFromSmiles(Chem.MolToSmiles(new_mol))

            if new_mol is None: continue
            
            has_error = False
            for nei_node in children:
                if nei_node.is_leaf: continue
                tmp_mol, tmp_mol2 = self.dfs_assemble(y_tree_mess, x_mol_vecs, all_nodes, cur_mol, new_global_amap, pred_amap, nei_node, cur_node, prob_decode, check_aroma, fast_uncertainty_decode=fast_uncertainty_decode)
                if tmp_mol is None: 
                    has_error = True
                    if i == 0: pre_mol = tmp_mol2
                    break
                cur_mol = tmp_mol

            if not has_error: return cur_mol, cur_mol

        return None, pre_mol
    
    def decoder_uncertainty_from_latent(self, z, num_sampled_models=10, num_sampled_outcomes=40, method="MI_Importance_sampling", return_SMILES=False, default_MI_value=np.inf):
        num_encoded_molecules = len(z) #z of dim (bs, latent_size * 2)
        z_tree, z_mol = z[:,:self.latent_size], z[:,self.latent_size:]
        self.train()
        if return_SMILES:
            decoded_smiles = []
        assert method in ['MI_Importance_sampling','NLL_prior'], "Not valid decoder uncertainty method"
        if method=="MI_Importance_sampling":
            ln_MI = torch.zeros((num_encoded_molecules, num_sampled_outcomes)).cuda()
            
            with torch.no_grad():
                for j in tqdm.tqdm(range(num_sampled_outcomes)):
                    decoded_smiles_iter = []
                    decoded_mol_trees_0 = []
                    ln_MI_j = torch.ones(num_encoded_molecules,dtype=torch.double).cuda() * math.log(default_MI_value) #Default value if all decoded trees are invalid
                    for z_idx in range(num_encoded_molecules):
                        smile_idx=self.decode(z_tree[z_idx].view(1, self.latent_size), z_mol[z_idx].view(1, self.latent_size), prob_decode=False, fast_uncertainty_decode=True)
                        smile_idx = Chem.MolToSmiles(Chem.MolFromSmiles(smile_idx), isomericSmiles=True)
                        decoded_smiles_iter.append(smile_idx)
                        if smile_idx is not None:
                            decoded_mol_trees_0.append(smiles_to_moltree(smile_idx))

                    none_smiles_mask = np.array([elem is None for elem in decoded_smiles_iter])
                    if len(decoded_mol_trees_0)==0:
                        ln_MI[:,j] = ln_MI_j
                        continue

                    if return_SMILES:
                        decoded_smiles.append(decoded_smiles_iter)
                    sampled_sequence_log_proba_across_models = torch.zeros((num_encoded_molecules, num_sampled_models), dtype=torch.float64).cuda()
                    try: 
                        tree_batch_0 = tensorize(decoded_mol_trees_0, self.vocab, assm=True)
                        tree_batch_0, x_jtenc_holder_0, x_mpn_holder_0, x_jtmpn_holder_0 = tree_batch_0
                        _, tree_mess_0 = self.jtnn(*x_jtenc_holder_0)
                    except:
                        ln_MI[:,j] = ln_MI_j
                        continue
                    
                    for t in range(num_sampled_models):
                        word_loss, topo_loss, word_acc, topo_acc = self.decoder(tree_batch_0, z_tree[~none_smiles_mask], mode="indep_loss")
                        assm_loss = self.assm(tree_batch_0, x_jtmpn_holder_0, z_mol[~none_smiles_mask], tree_mess_0, mode="indep_loss")
                        model_sequence_log_proba = word_loss + topo_loss + assm_loss
                        sampled_sequence_log_proba_across_models[:,t] = - model_sequence_log_proba
                    log_p_tot_j = - torch.log(torch.tensor(num_sampled_models).float()) + sampled_sequence_log_proba_across_models.logsumexp(dim=1)
                
                    ln_B = log_p_tot_j + torch.log( - log_p_tot_j)
                    
                    pj_logpj = sampled_sequence_log_proba_across_models + torch.log(- sampled_sequence_log_proba_across_models)
                    ln_A = - torch.log(torch.tensor(num_sampled_models).float()) + pj_logpj.logsumexp(dim=1)
                    
                    ln_MI_j[~none_smiles_mask] = ou.LDE(ln_B,ln_A) - log_p_tot_j
                    ln_MI[:,j] = ln_MI_j

                ln_MI_avg = - torch.log(torch.tensor(num_sampled_outcomes).float()) + ln_MI.logsumexp(dim=1)
                
                if return_SMILES:
                    decoded_smiles = np.array(decoded_smiles) # (num_sampled_outcomes, latent_size)
                    return torch.exp(ln_MI_avg.squeeze()), np.transpose(decoded_smiles)
                else:
                    return torch.exp(ln_MI_avg.squeeze())
        
        elif method == "NLL_prior":
            m = rnv.MultivariateNormal(torch.zeros(self.latent_size * 2).cuda(), torch.eye(self.latent_size * 2).cuda())
            return - m.log_prob(z)
            
    def log_proba_under_prior(self,z):
        m = rnv.MultivariateNormal(torch.zeros(self.latent_size * 2).cuda(), torch.eye(self.latent_size * 2).cuda())
        return m.log_prob(z) 
    

class JTNNVAE_prop(JTNNVAE):
    
    def __init__(self, vocab, hidden_size, latent_size, depthT, depthG, prop, dropout_rate_GRU=0.0, dropout_rate_MLP=0.0, drop_prop_NN=0.0):
        super(JTNNVAE_prop, self).__init__(vocab, hidden_size, latent_size, depthT, depthG, dropout_rate_GRU, dropout_rate_MLP)
        self.prop = prop
        self.drop_prop_NN = drop_prop_NN
        self.prop_loss = nn.MSELoss()
        self.prop_net = nn.Sequential(
                    nn.Dropout(drop_prop_NN),
                    nn.Linear(self.latent_size * 2, self.hidden_size),
                    nn.Dropout(drop_prop_NN),
                    nn.Tanh(),
                    nn.Linear(self.hidden_size, 1)
            )
    
    def save_params(self, save_path, name_parameter_file):
        params = {
            'hidden_size': self.hidden_size, 
            'latent_size': int(self.latent_size * 2),
            'depthT': self.depthT, 
            'depthG': self.depthG, 
            'prop': self.prop,
            'dropout_rate_GRU': self.dropout_rate_GRU,
            'dropout_rate_MLP': self.dropout_rate_MLP,
            'drop_prop_NN': self.drop_prop_NN
        }
        with open(save_path + os.sep + name_parameter_file, 'w') as fp:
            json.dump(params, fp)
    
    def forward(self, x_batch, beta):
        x_batch, prop_batch = list(zip(*x_batch))
        
        x_batch, x_jtenc_holder, x_mpn_holder, x_jtmpn_holder = tensorize(x_batch, self.vocab, assm=True)
        x_tree_vecs, x_tree_mess, x_mol_vecs = self.encode(x_jtenc_holder, x_mpn_holder)
        z_tree_vecs,tree_kl = self.rsample(x_tree_vecs, self.T_mean, self.T_var)
        z_mol_vecs,mol_kl = self.rsample(x_mol_vecs, self.G_mean, self.G_var)

        all_vec = torch.cat([z_tree_vecs, z_mol_vecs], dim=1)
        prop_label = torch.Tensor(prop_batch).cuda()
        prop_loss = self.prop_loss(self.prop_net(all_vec).squeeze(), prop_label)

        kl_div = tree_kl + mol_kl
        word_loss, topo_loss, word_acc, topo_acc = self.decoder(x_batch, z_tree_vecs)
        assm_loss, assm_acc = self.assm(x_batch, x_jtmpn_holder, z_mol_vecs, x_tree_mess)

        loss = word_loss + topo_loss + assm_loss + beta * kl_div + prop_loss
        return loss, kl_div.item(), word_acc, topo_acc, assm_acc, prop_loss.item()
    
    def property_prediction_uncertainty_from_latent(self, z, num_samples_uncertainty=1000):
        self.train() #Move to train mode to ensure dropout is turned on

        z_batch = torch.repeat_interleave(z, num_samples_uncertainty, dim=0)
        property_predictions = self.prop_net.forward(z_batch).view(-1, num_samples_uncertainty) #(bs, num_samples_uncertainty)
            
        avg_property_predictions = torch.mean(property_predictions, dim=1, keepdim=False)
        uncertainty_property_predictions = torch.std(property_predictions, dim=1, keepdim=False)

        return avg_property_predictions, uncertainty_property_predictions