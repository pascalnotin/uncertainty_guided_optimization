import sys
import argparse
import rdkit
import rdkit.Chem as Chem

from JTVAE.fast_jtnn.chemutils import get_clique_mol, tree_decomp, get_mol, get_smiles, set_atommap, enum_assemble, decode_stereo
from JTVAE.fast_jtnn.vocab import *

class MolTreeNode(object):

    def __init__(self, smiles, clique=[]):
        self.smiles = smiles
        self.mol = get_mol(self.smiles)

        self.clique = [x for x in clique]
        self.neighbors = []
        
    def add_neighbor(self, nei_node):
        self.neighbors.append(nei_node)

    def recover(self, original_mol):
        clique = []
        clique.extend(self.clique)
        if not self.is_leaf:
            for cidx in self.clique:
                original_mol.GetAtomWithIdx(cidx).SetAtomMapNum(self.nid)

        for nei_node in self.neighbors:
            clique.extend(nei_node.clique)
            if nei_node.is_leaf:
                continue
            for cidx in nei_node.clique:
                if cidx not in self.clique or len(nei_node.clique) == 1:
                    atom = original_mol.GetAtomWithIdx(cidx)
                    atom.SetAtomMapNum(nei_node.nid)

        clique = list(set(clique))
        label_mol = get_clique_mol(original_mol, clique)
        self.label = Chem.MolToSmiles(Chem.MolFromSmiles(get_smiles(label_mol)))

        for cidx in clique:
            original_mol.GetAtomWithIdx(cidx).SetAtomMapNum(0)

        return self.label 
    
    def assemble(self):
        neighbors = [nei for nei in self.neighbors if nei.mol.GetNumAtoms() > 1]
        neighbors = sorted(neighbors, key=lambda x:x.mol.GetNumAtoms(), reverse=True)
        singletons = [nei for nei in self.neighbors if nei.mol.GetNumAtoms() == 1]
        neighbors = singletons + neighbors

        cands,aroma = enum_assemble(self, neighbors)
        new_cands = [cand for i,cand in enumerate(cands) if aroma[i] >= 0]
        if len(new_cands) > 0: cands = new_cands

        if len(cands) > 0:
            self.cands, _ = list(zip(*cands))
            self.cands = list(self.cands)
        else:
            self.cands = []
        
class MolTree(object):

    def __init__(self, smiles):
        self.smiles = smiles
        self.mol = get_mol(smiles)
        
        #Stereo Generation (currently disabled)
        #mol = Chem.MolFromSmiles(smiles)
        #self.smiles3D = Chem.MolToSmiles(mol, isomericSmiles=True)
        #self.smiles2D = Chem.MolToSmiles(mol)
        #self.stereo_cands = decode_stereo(self.smiles2D)

        cliques, edges = tree_decomp(self.mol)  
        self.nodes = []
        root = 0
        for i,c in enumerate(cliques):
            cmol = get_clique_mol(self.mol, c)
            node = MolTreeNode(get_smiles(cmol), c)
            self.nodes.append(node)
            if min(c) == 0: root = i

        for x,y in edges:
            self.nodes[x].add_neighbor(self.nodes[y])
            self.nodes[y].add_neighbor(self.nodes[x])
        
        if root > 0:
            self.nodes[0],self.nodes[root] = self.nodes[root],self.nodes[0]

        for i,node in enumerate(self.nodes):
            node.nid = i + 1
            if len(node.neighbors) > 1:
                set_atommap(node.mol, node.nid)
            node.is_leaf = (len(node.neighbors) == 1)

    def size(self):
        return len(self.nodes)

    def recover(self):
        for node in self.nodes:
            node.recover(self.mol)

    def assemble(self):
        for node in self.nodes:
            node.assemble()

def dfs(node, fa_idx):
    max_depth = 0
    for child in node.neighbors:
        if child.idx == fa_idx:
            continue
        max_depth = max(max_depth, dfs(child, node.idx))
    return max_depth + 1

if __name__ == "__main__":
    """
    The main script below may be used to generate a new vocabulary on new data
    """
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', default=None, type=str, help='Path to input data folder')
    parser.add_argument('--vocab_location', default=None, type=str, help='Path where to store generated vocabulary')
    args = parser.parse_args()

    MAX_TREE_WIDTH=50
    cset = set()
    with open(args.data_folder, 'r') as input_file:
        for i, line in enumerate(input_file.readlines()):
            smiles = line.strip().split()[0]
            alert = False
            mol = MolTree(smiles)
            for c in mol.nodes:
                if c.mol.GetNumAtoms() > MAX_TREE_WIDTH:
                    alert = True
                cset.add(c.smiles)
            if len(mol.nodes) > 1 and alert:
                sys.stderr.write('[WARNING]: %d-th molecule %s has a high tree-width.\n' % (i + 1, smiles))
    
    with open(args.vocab_location, 'w') as vocab_file:
        for x in cset:
            vocab_file.write(x+'\n')