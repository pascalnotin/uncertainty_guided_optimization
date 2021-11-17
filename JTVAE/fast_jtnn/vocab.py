import os
from pathlib import Path
import copy
import rdkit
import rdkit.Chem as Chem

path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)
parent_dir = Path(dir_path).parent.absolute()

def get_slots(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return [(atom.GetSymbol(), atom.GetFormalCharge(), atom.GetTotalNumHs()) for atom in mol.GetAtoms()]

class Vocab(object):
    benzynes = ['C1=CC=CC=C1', 'C1=CC=NC=C1', 'C1=CC=NN=C1', 'C1=CN=CC=N1', 'C1=CN=CN=C1', 'C1=CN=NC=N1', 'C1=CN=NN=C1', 'C1=NC=NC=N1', 'C1=NN=CN=N1']
    penzynes = ['C1=C[NH]C=C1', 'C1=C[NH]C=N1', 'C1=C[NH]N=C1', 'C1=C[NH]N=N1', 'C1=COC=C1', 'C1=COC=N1', 'C1=CON=C1', 'C1=CSC=C1', 'C1=CSC=N1', 'C1=CSN=C1', 'C1=CSN=N1', 'C1=NN=C[NH]1', 'C1=NN=CO1', 'C1=NN=CS1', 'C1=N[NH]C=N1', 'C1=N[NH]N=C1', 'C1=N[NH]N=N1', 'C1=NN=N[NH]1', 'C1=NN=NS1', 'C1=NOC=N1', 'C1=NON=C1', 'C1=NSC=N1', 'C1=NSN=C1']

    def __init__(self, smiles_list):
        self.vocab = smiles_list
        self.vmap = {x:i for i,x in enumerate(self.vocab)}
        self.slots = [get_slots(smiles) for smiles in self.vocab]
        Vocab.benzynes = [s for s in smiles_list if s.count('=') >= 2 and Chem.MolFromSmiles(s).GetNumAtoms() == 6] + ['C1=CCNCC1']
        Vocab.penzynes = [s for s in smiles_list if s.count('=') >= 2 and Chem.MolFromSmiles(s).GetNumAtoms() == 5] + ['C1=NCCN1','C1=NNCC1']
        
    def get_index(self, smiles):
        try:
            return self.vmap[smiles]
        except:
            print("Following smiles not in vocab: "+str(smiles))
            with open(str(parent_dir)+os.sep+'data'+os.sep+'zinc'+os.sep+'additional_vocab_terms.txt', "a+") as logs_file:
                logs_file.write(smiles+"\n")
            return 0

    def get_smiles(self, idx):
        return self.vocab[idx]

    def get_slots(self, idx):
        return copy.deepcopy(self.slots[idx])

    def size(self):
        return len(self.vocab)

