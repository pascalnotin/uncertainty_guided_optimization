import math, random, sys, os
import numpy as np
import pandas as pd
import networkx as nx
import func_timeout
import tqdm
from rdkit import RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem import rdmolops
import rdkit.Chem.QED

import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import UpperConfidenceBound,ExpectedImprovement,ProbabilityOfImprovement, qExpectedImprovement, qUpperConfidenceBound, qNoisyExpectedImprovement
from botorch.optim import optimize_acqf

from utils import sascorer, quality_filters as qual

lg = RDLogger.logger() 
lg.setLevel(RDLogger.CRITICAL)

path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

###################################### NUMERICAL STABILITY ##################################################
def LSE(log_ai):
    """
    log_ai of dim (batch_size, num_items_in_sum)
    """
    max_log,_ = log_ai.max(dim=1,keepdim=True)
    return max_log.squeeze() + torch.log(torch.exp(log_ai - max_log).sum(dim=1,keepdim=False))

def LDE(log_ai,log_bi):
    max_log_p = torch.max(log_ai,log_bi)
    min_log_p = torch.min(log_ai,log_bi)
    return (max_log_p + torch.log(1 - torch.exp(min_log_p - max_log_p)))

###################################### MOLECULE PROPERTIES ##################################################
logP_file = dir_path + os.sep + 'stats_training_data/logP_values.txt'
SAS_file = dir_path + os.sep + 'stats_training_data/SA_scores.txt'
cycle_file = dir_path + os.sep + 'stats_training_data/cycle_scores.txt'

logP_values = np.loadtxt(logP_file)
SAS_values = np.loadtxt(SAS_file)
cycle_values = np.loadtxt(cycle_file)

training_stats = {
'logP_mean':np.mean(logP_values),
'logP_std':np.std(logP_values),
'SAS_mean':np.mean(SAS_values),
'SAS_std':np.std(SAS_values),
'cycles_mean':np.mean(cycle_values),
'cycles_std':np.std(cycle_values)
}

#Property stats training data
final_logP_train_stats_raw={'mean': -0.002467457978476197, 'std': 2.056736565112327, 'median': 0.42761702630532883, 'min': -62.516944569759666, 'max': 4.519902819580757, 'P1': -6.308202037634639, 'P5': -3.7061575195672125, 'P10': -2.6097184083169522, 'P25': -1.0492552134450062, 'P75': 1.4174359964331003, 'P90': 2.1113332292393188, 'P95': 2.4569317747277495, 'P99': 3.0048043651582605}
final_logP_train_stats_normalized={'mean': -0.0013269769793680093, 'std': 1.0022175676799359, 'median': 0.20822120507327543, 'min': -30.46370322413232, 'max': 2.2023601097894416, 'P1': -3.0740150902231402, 'P5': -1.8060773166698125, 'P10': -1.2717987692036161, 'P25': -0.5114081551001504, 'P75': 0.6905739551134478, 'P90': 1.0286998043562519, 'P95': 1.1971048594070872, 'P99': 1.464075062137245}

#Decoder uncertainty stats
decoder_uncertainty_stats_training ={
    'JTVAE': {
        'MI_Importance_sampling': {'mean': 0.7737001577503979, 'std': 0.7191886465214079, 'median': 0.6115016341209412, 'min': 0.003500204300507903, 'max': 3.2164592742919917, 'P1': 0.004812391460873187, 'P5': 0.03621037751436234, 'P25': 0.16248027607798576, 'P75': 1.1251116693019867, 'P95': 2.4251182436943055, 'P99': 2.9005215597152705},
        'NLL_prior': {'mean': 110.49043981933593, 'std': 22.952045705008157, 'median': 106.87257385253906, 'min': 80.51214599609375, 'max': 199.3219451904297, 'P1': 83.63397506713868, 'P5': 86.59568367004395, 'P25': 96.7742748260498, 'P75': 117.61268424987794, 'P95': 147.31882400512683, 'P99': 195.5686897277832}
    }
}


def verify_smile(smile):
        return (smile != '') and pd.notnull(smile) and (rdkit.Chem.MolFromSmiles(smile) is not None)

def clean_up_smiles(smiles):
    return list(map(lambda x: x.strip(), smiles))

def compute_qed(smile, default_value=np.nan):
    try:
        mol= rdkit.Chem.MolFromSmiles(smile)
        qed = rdkit.Chem.QED.qed(mol)
        return qed
    except:
        return default_value

def compute_sas(smile, default_value=np.nan):
    try: 
        mol = rdkit.Chem.MolFromSmiles(smile)
        sas = sascorer.calculateScore(mol)
        return sas
    except: 
        return default_value

def compute_logP(smile, default_value=np.nan):
    try: 
        mol = rdkit.Chem.MolFromSmiles(smile)
        logp = Descriptors.MolLogP(mol)
        return logp
    except: 
        return default_value

def compute_logPminusSAS_score(smile, default_value=np.nan):
    try: 
        mol = rdkit.Chem.MolFromSmiles(smile)
        score = Descriptors.MolLogP(mol) - sascorer.calculateScore(mol)
        return score
    except: 
        return default_value

def compute_target_logP(smile, default_value=np.nan, train_stats = training_stats):
    try: 
        mol = rdkit.Chem.MolFromSmiles(smile)
        
        logP_score = Descriptors.MolLogP(mol)
        
        SAS_score =  - sascorer.calculateScore(mol)
        
        cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(mol)))
        if len(cycle_list) == 0:
            cycle_length = 0
        else:
            cycle_length = max([len(j) for j in cycle_list])
        if cycle_length <= 6:
            cycle_length = 0
        else:
            cycle_length = cycle_length - 6
        
        cycle_score = - cycle_length

        logP_score_normalized = (logP_score - train_stats['logP_mean']) / train_stats['logP_std']
        SAS_score_normalized = (SAS_score - train_stats['SAS_mean']) / train_stats['SAS_std']
        cycle_score_normalized = (cycle_score - train_stats['cycles_mean']) / train_stats['cycles_std']

        return logP_score_normalized + SAS_score_normalized + cycle_score_normalized
    except: 
        return default_value

def convert_tensors_to_smiles(tensor_molecules, indices_chars):
    """
    For CharVAE only. Input tensor_molecules of size (batch, seq_len, n_chars)
    """
    smiles = []
    for molecule in tensor_molecules.detach().cpu().numpy():
        temp_str = ""
        for atom_j in molecule:
            index = np.argmax(atom_j) 
            temp_str += indices_chars[index]
        smiles.append(temp_str)
    return np.array(clean_up_smiles(smiles))

def compute_stats(input_array, mode="nan"):
    if mode =="normal":
        return {
            'mean':input_array.mean(),
            'std':input_array.std(),
            'median':np.median(input_array),
            'min':input_array.min(),
            'max':input_array.max(),
            'P1':np.percentile(input_array,1),
            'P5':np.percentile(input_array,5),
            'P25':np.percentile(input_array,25),
            'P75':np.percentile(input_array,75),
            'P95':np.percentile(input_array,95),
            'P99':np.percentile(input_array,99)
        }
    elif mode=="nan":
        return {
            'mean':np.nanmean(input_array),
            'std': np.nanstd(input_array),
            'median':np.nanmedian(input_array),
            'min':np.nanmin(input_array),
            'max':np.nanmax(input_array),
            'P1': np.nanpercentile(input_array,1),
            'P5': np.nanpercentile(input_array,5),
            'P25':np.nanpercentile(input_array,25),
            'P75':np.nanpercentile(input_array,75),
            'P95':np.nanpercentile(input_array,95),
            'P99':np.nanpercentile(input_array,99)
        }


def check_validity_objects(smiles, return_valid=True):
    """smiles is a list of molecule SMILE representation.
    Returns valid smiles generated by default, since needed to compute unicity and novelty."""
    num_molecules=len(smiles)
    if num_molecules==0:
        print("No valid modelcule generated!")
        return 0
    valid_smiles=[]
    for smile in smiles:
        if verify_smile(smile):
            valid_smiles.append(smile)
    if return_valid:
        return len(valid_smiles) / float(num_molecules), valid_smiles
    else:
        return len(valid_smiles) / float(num_molecules)

def check_unicity_objects(smiles):
    """Need to pass in valid smiles"""
    unique_smiles = set() #empty set
    num_molecules=len(smiles)
    if num_molecules==0:
        return 0
    else:
        for smile in smiles:
            if smile not in unique_smiles:
                unique_smiles.add(smile)
        return len(unique_smiles)/float(num_molecules)

def check_novelty_objects(smiles,training_smiles, verbose=False):
    """Need to pass in valid smiles"""
    count_in_training=0
    num_molecules_generated = len(smiles)
    if num_molecules_generated==0:
        return 0
    else:
        training_smiles_set=set(training_smiles)
        num_molecules_training = len(training_smiles_set)
        if verbose:
            print("Num distinct molecules in training data: "+str(num_molecules_training))
        for smile in smiles:
            if smile in training_smiles_set:
                count_in_training+=1
        if verbose:
            print("Num generated molecules that were already in training data: "+str(count_in_training))
        return 1 - count_in_training/float(num_molecules_generated)

def log_stats(file_name, stats, log_entry):
    with open(file_name, "a+") as logs_file:
        logs_file.write(log_entry+"\n")
        logs_file.write(str(stats))
        logs_file.write("\n\n")

class assessment_generated_objects():
    
    def __init__(self, generated_objects_list, model_training_data, prop="final_logP"):
        """Function that returns the property of the best 3 objects generated; top 50 (or less if fewer valid) and average over all generated
        Also return % valid, %unique, %novel of all generated elements"""
        
        self.num_generated_objects = len(generated_objects_list)
        
        #Compute Validity of generated objects
        self.validity_all, self.valid_generated_objects_list = check_validity_objects(generated_objects_list, return_valid=True)
        self.num_valid_generated_objects = len(self.valid_generated_objects_list)
        
        #Compute Properties of generated objects
        self.property_generated_objects = []
        if prop=="QED":
            for valid_generated_object in self.valid_generated_objects_list:
                self.property_generated_objects.append(compute_qed(valid_generated_object))
        elif prop=="logPminusSAS":
            for valid_generated_object in self.valid_generated_objects_list:
                self.property_generated_objects.append(compute_logPminusSAS_score(valid_generated_object))
        elif prop=="final_logP":
            for valid_generated_object in self.valid_generated_objects_list:
                self.property_generated_objects.append(compute_target_logP(valid_generated_object))
        
        try:
            self.stats_property_generated_objects = compute_stats(np.array(self.property_generated_objects))
        except:
            self.stats_property_generated_objects = None
        
        property_df = pd.DataFrame({'Valid_generated_objects': np.array(self.valid_generated_objects_list),
                                    'Property_valid_generated_objects': np.array(self.property_generated_objects)})
        
        #quality_filters
        if len(property_df)>0:
            QF = qual.QualityFiltersCheck(training_data_smi=[])
            property_df['Pass_quality_filters']= QF.check_smiles_pass_quality_filters_flag(self.valid_generated_objects_list).astype(bool)
        
        property_df.sort_values(by=['Property_valid_generated_objects'], ascending=False, inplace=True)
        
        #De-normalize scores
        if prop=="final_logP":
            property_df['Property_valid_generated_objects'] = property_df['Property_valid_generated_objects']*final_logP_train_stats_raw['std'] + final_logP_train_stats_raw['mean']
        
        property_df.reset_index(inplace=True, drop=True)
        self.top10_valid_molecules  = property_df['Valid_generated_objects'][:10]
        self.top50_valid_molecules  = property_df['Valid_generated_objects'][:50]

        self.top_properties_scores={}
        self.top_properties_smiles={}
        self.len_property_df = len(property_df)
        for i in range(1,11):
            if  self.len_property_df > i-1:
                self.top_properties_scores['top_'+str(i)] = property_df['Property_valid_generated_objects'][i-1]
                self.top_properties_smiles['top_'+str(i)] = property_df['Valid_generated_objects'][i-1]
            else:
                self.top_properties_scores['top_'+str(i)] = None
                self.top_properties_smiles['top_'+str(i)] = None

        if self.len_property_df > 0:
            #Avg property
            self.property_all = property_df['Property_valid_generated_objects'].mean()
            self.property_top10 = property_df['Property_valid_generated_objects'][:10].mean()
            self.property_top50 = property_df['Property_valid_generated_objects'][:50].mean()

            #Compute Unicity of generated objects
            self.unicity_all = check_unicity_objects(self.valid_generated_objects_list)
            self.unicity_top10 = check_unicity_objects(self.top10_valid_molecules)
            
            #Compute Novelty of generated objects
            self.novelty_all = check_novelty_objects(self.valid_generated_objects_list, model_training_data)
            self.novelty_top10 = check_novelty_objects(self.top10_valid_molecules, model_training_data)

            #Quality
            self.quality_all = property_df['Pass_quality_filters'].astype(int).mean()
            self.quality_top10 = np.nanmean(QF.check_smiles_pass_quality_filters_flag(self.top10_valid_molecules))

            #QED
            self.qed_all = np.nanmean(np.array([compute_qed(x) for x in self.valid_generated_objects_list]))
            self.qed_top10 = np.nanmean(np.array([compute_qed(x) for x in self.top10_valid_molecules]))
        else:
            self.property_all = None
            self.property_top10 = None
            self.property_top50 = None
            self.unicity_all = None
            self.unicity_top10 = None
            self.novelty_all = None
            self.novelty_top10 = None
            self.quality_all = None
            self.quality_top10 = None
            self.qed_all = None
            self.qed_top10 = None
        
        #Stats passing quality filters
        property_df_qual = property_df[property_df['Pass_quality_filters']]
        property_df_qual.reset_index(inplace=True)
        if len(property_df_qual)>0:
            self.property_all_qual = property_df_qual['Property_valid_generated_objects'].mean()
            self.property_top5avg_qual = property_df_qual['Property_valid_generated_objects'][:5].mean()
            self.property_top10avg_qual = property_df_qual['Property_valid_generated_objects'][:10].mean()
            self.property_top50avg_qual = property_df_qual['Property_valid_generated_objects'][:50].mean()
            self.qed_all_qual = np.nanmean(np.array([compute_qed(x) for x in property_df_qual['Valid_generated_objects']]))
            self.qed_top10_qual = np.nanmean(np.array([compute_qed(x) for x in property_df_qual['Valid_generated_objects'][:10]]))
        else:
            self.property_all_qual = None
            self.property_top5avg_qual = None
            self.property_top10avg_qual = None
            self.property_top50avg_qual = None
            self.qed_all_qual = None
            self.qed_top10_qual = None
        
        self.property_top_qual = {}
        for i in range(1,11):
            if len(property_df_qual) > i-1:
                self.property_top_qual[i]=property_df_qual['Property_valid_generated_objects'][i-1]
            else:
                self.property_top_qual[i]=None        
    
    def log_all_stats_generated_objects(self, filename):
        results={}
        log_stats(file_name= filename, stats=self.num_generated_objects, log_entry="Number of generated objects")
        log_stats(file_name= filename, stats=self.validity_all, log_entry="Proportion of valid generated objects")
        results['validity_all']=self.validity_all
        log_stats(file_name= filename, stats=self.unicity_all, log_entry="Proportion of unique valid generated objects")
        results['unicity_all']=self.unicity_all
        results['unicity_top10']=self.unicity_top10
        log_stats(file_name= filename, stats=self.novelty_all, log_entry="Proportion of novel valid generated objects")
        results['novelty_all']=self.novelty_all
        results['novelty_top10']=self.novelty_top10
        log_stats(file_name= filename, stats=self.quality_all, log_entry="Proportion of valid generated objects passing quality filters")
        results['quality_all']=self.quality_all
        results['quality_top10']=self.quality_top10
        log_stats(file_name= filename, stats=self.qed_all, log_entry="Avg qed of valid generated objects")
        results['qed_all']=self.qed_all
        results['qed_top10']=self.qed_top10
        log_stats(file_name= filename, stats=self.stats_property_generated_objects, log_entry="Stats of properties of generated objects")
        results['target_property_all']=self.property_all
        results['target_property_top10']=self.property_top10
        results['target_property_top50']=self.property_top50

        for i in range(1,11):
            if  self.len_property_df > i-1:
                log_stats(file_name= filename, stats=self.top_properties_scores['top_'+str(i)], log_entry="Property of top "+str(i)+" generated object")
                log_stats(file_name= filename, stats=self.top_properties_smiles['top_'+str(i)], log_entry="Smiles of top "+str(i)+" generated object")
                results['top'+str(i)]=self.top_properties_scores['top_'+str(i)]
            else:
                results['top'+str(i)]=None

        #Qual metrics
        results['property_all_qual'] = self.property_all_qual
        for i in range(1,11):
            results['property_top'+str(i)+'_qual'] = self.property_top_qual[i]
        results['property_top5avg_qual'] = self.property_top5avg_qual
        results['property_top10avg_qual'] = self.property_top10avg_qual
        results['property_top50avg_qual'] = self.property_top50avg_qual
        results['qed_all_qual'] = self.qed_all_qual
        results['qed_top10_qual'] = self.qed_top10_qual
        
        results['top_10_molecules'] = self.top10_valid_molecules
        return results
        

###################################### OPTIMIZATION INITIALIZATION ##################################################

def starting_objects_latent_embeddings(model, data, mode="random", num_objects_to_select=100, batch_size=256, property_upper_bound=None, model_type="JTVAE"):
    if model_type=="JTVAE":
        latent_space_dim = model.latent_size * 2
    elif model_type=="CharVAE":
        latent_space_dim = model.params.z_dim
        
    if mode=="random":
        num_objects_data = len(data)
        selected_objects_indices = np.random.choice(a=range(num_objects_data), size=num_objects_to_select, replace=False).tolist()
        starting_objects = np.array(data)[selected_objects_indices]
        if model_type=="JTVAE":
            starting_objects_smiles = starting_objects
        elif model_type=="CharVAE":
            starting_objects_smiles = convert_tensors_to_smiles(starting_objects, model.params.indices_char)
        starting_objects_latent_embeddings = torch.zeros(num_objects_to_select, latent_space_dim).to(device)
        starting_objects_properties = []
        for batch_object_indices in range(0,num_objects_to_select,batch_size):
            a, b = batch_object_indices, batch_object_indices+batch_size
            if model_type=="JTVAE":
                starting_objects_latent_embeddings[a:b] = model.encode_and_samples_from_smiles(starting_objects[a:b])
            elif model_type=="CharVAE":
                mu, log_var = model.encoder(starting_objects[a:b])
                starting_objects_latent_embeddings[a:b] = model.sampling(mu, log_var)
            for smile in starting_objects_smiles[a:b]:
                starting_objects_properties.append(compute_target_logP(smile))
        starting_objects_properties = torch.tensor(starting_objects_properties)

    elif mode=="low_property_objects":
        num_starting_points_selected = 0
        index_object_in_dataset = 0
        starting_objects = []
        starting_objects_smiles = []
        starting_objects_properties = []
        starting_objects_latent_embeddings = torch.zeros(num_objects_to_select, latent_space_dim).to(device)

        while num_starting_points_selected < num_objects_to_select:
            if model_type=='JTVAE':
                smile_potential_starting_object = data[index_object_in_dataset]
            elif model_type=='CharVAE':
                potential_starting_object = data[index_object_in_dataset].unsqueeze(0)
                smile_potential_starting_object = convert_tensors_to_smiles(potential_starting_object, model.params.indices_char)[0]
            final_logP = compute_target_logP(smile_potential_starting_object)
            if final_logP < property_upper_bound and final_logP > - 100:
                if model_type=='JTVAE':
                    new_object_latent_representation = model.encode_and_samples_from_smiles([smile_potential_starting_object])
                elif model_type=='CharVAE':
                    mu, log_var = model.encoder(potential_starting_object)
                    new_object_latent_representation  = model.sampling(mu, log_var)
                starting_objects_latent_embeddings[num_starting_points_selected] = new_object_latent_representation       
                starting_objects_properties.append(final_logP)
                starting_objects_smiles.append(smile_potential_starting_object)
                num_starting_points_selected+=1
            index_object_in_dataset+=1
        starting_objects_properties=torch.tensor(starting_objects_properties)
    
    return starting_objects_latent_embeddings, starting_objects_properties, starting_objects_smiles

###################################### OPTIMIZATION ROUTINES ##################################################

def gradient_ascent_optimization(model, starting_objects_latent_embeddings, number_gradient_steps=10, 
                                uncertainty_decoder_method=None, num_sampled_models=10, num_sampled_outcomes=40,
                                model_decoding_mode=None, model_decoding_topk_value=None, alpha=1.0, normalize_gradients=True, 
                                batch_size=64, uncertainty_threshold="No_constraint", keep_all_generated=False, model_type="JTVAE"
                                ):
    """
    Perform gradient ascent in latent space. Filter out invalid points, ie. above uncertainty threshold. Keep last number_starting_objects valid points.
    model_decoding_mode and model_decoding_topk_value are only relevant for RNN decoding (CharVAE).
    """
    number_starting_objects = len(starting_objects_latent_embeddings)
    generated_objects_list=[]
    if model_type=='JTVAE':
        hidden_dim = model.latent_size*2
    elif model_type=='CharVAE':
        hidden_dim = model.params.z_dim
        if model_decoding_mode is not None:
            model.sampling_mode = model_decoding_mode
            model.generation_top_k_sampling = model_decoding_topk_value
    
    if uncertainty_threshold!='No_constraint':
        uncertainty_threshold_value = decoder_uncertainty_stats_training[model_type][uncertainty_decoder_method][uncertainty_threshold]
    
    all_points_latent_representation = torch.zeros((number_gradient_steps+1)*number_starting_objects, hidden_dim)
    all_points_latent_representation[:number_starting_objects] = starting_objects_latent_embeddings.view(-1,hidden_dim)
    
    new_objects_latent_representation = starting_objects_latent_embeddings
    for step in tqdm.tqdm(range(1, number_gradient_steps+1)):
        torch.cuda.empty_cache()
        model.zero_grad()
        gradient = torch.zeros(number_starting_objects, hidden_dim).to(device)
        for batch_object_indices in range(0, number_starting_objects, batch_size):
            model.zero_grad()
            a, b = batch_object_indices , batch_object_indices+batch_size
            new_objects_latent_representation_slice = torch.autograd.Variable(new_objects_latent_representation[a:b], requires_grad=True)
            if model_type=='JTVAE':
                predicted_property_slice = model.prop_net(new_objects_latent_representation_slice).squeeze()
                predicted_property_slice = (predicted_property_slice - (final_logP_train_stats_raw['mean'])) / (final_logP_train_stats_raw['std'])
            elif model_type=='CharVAE':
                predicted_property_slice = model.qed_net(new_objects_latent_representation_slice).squeeze()
            
            gradient[a:b] = torch.autograd.grad(outputs = predicted_property_slice,
                                                inputs = new_objects_latent_representation_slice,
                                                grad_outputs = torch.ones_like(predicted_property_slice).to(device), 
                                                retain_graph=False)[0]
        if normalize_gradients:
            gradient /= torch.norm(gradient,2)
        
        new_objects_latent_representation = new_objects_latent_representation + alpha * gradient
        all_points_latent_representation[step*number_starting_objects:(step+1)*number_starting_objects] = new_objects_latent_representation.view(-1,hidden_dim)

    if uncertainty_threshold!='No_constraint':
        if keep_all_generated: #Need to compute uncertainty for all points
            with torch.no_grad():
                num_points_total  = (number_gradient_steps+1)*number_starting_objects
                uncertainty_all_points = torch.zeros(num_points_total)
                for batch_object_indices in range(0, num_points_total, batch_size):
                    z_slice = all_points_latent_representation[batch_object_indices:batch_object_indices+batch_size].to(device)
                    uncertainty_all_points[batch_object_indices:batch_object_indices+batch_size] = model.decoder_uncertainty_from_latent(
                                                                                                    z = z_slice,
                                                                                                    method = uncertainty_decoder_method,
                                                                                                    num_sampled_models=num_sampled_models, 
                                                                                                    num_sampled_outcomes=num_sampled_outcomes
                                                                                                    ).squeeze().detach().cpu()
                index_below_uncertainty_threshold = (uncertainty_all_points < uncertainty_threshold_value)
                all_points_latent_representation = all_points_latent_representation[index_below_uncertainty_threshold]
                selected_points = all_points_latent_representation
        else: #We compute uncertainty in batches starting from latest batch of points generated, and continue until we have reached the desired number of points below uncertainty threshold
            with torch.no_grad():
                num_points_to_generate = number_starting_objects
                point_index  = (number_gradient_steps+1)*number_starting_objects + 1
                selected_points=[]
                while num_points_to_generate > 0:
                    if point_index>0:
                        potential_points=all_points_latent_representation[max(point_index-batch_size,0):point_index].view(-1,hidden_dim).to(device)
                        uncertainty_potential_points = model.decoder_uncertainty_from_latent(
                                                                                            z = potential_points,
                                                                                            method = uncertainty_decoder_method,
                                                                                            num_sampled_models=num_sampled_models, 
                                                                                            num_sampled_outcomes=num_sampled_outcomes
                                                                                            ).squeeze().detach().cpu()
                        count_below=(uncertainty_potential_points < uncertainty_threshold_value).sum()
                        num_points_to_generate -=count_below
                        selected_points.extend(potential_points[uncertainty_potential_points < uncertainty_threshold_value])
                        point_index-=batch_size
                        
                selected_points=selected_points[:number_starting_objects]
    else:
        if keep_all_generated:
            selected_points=all_points_latent_representation
        else:
            selected_points=all_points_latent_representation[-number_starting_objects:]

    with torch.no_grad():
        if model_type=='JTVAE':
            for idx in range(len(selected_points)):
                z = selected_points[idx].view(1,hidden_dim).to(device)
                z_tree, z_mol = z[:,:model.latent_size], z[:,model.latent_size:]
                smiles_new_objects =  model.decode(z_tree, z_mol, prob_decode=False)
                generated_objects_list.append(smiles_new_objects)
        elif model_type=='CharVAE':
            for batch_object_indices in range(0, len(selected_points), batch_size):
                decoded_new_objects =  model.generate_from_latent(selected_points[batch_object_indices:batch_object_indices+batch_size].to(device))
                smiles_new_objects = convert_tensors_to_smiles(decoded_new_objects, model.params.indices_char)
                generated_objects_list.append(smiles_new_objects)

    return generated_objects_list

def bayesian_optimization(model, starting_objects_latent_embeddings, starting_objects_properties, number_BO_steps, BO_uncertainty_mode, 
                            BO_uncertainty_threshold='No_constraint', BO_uncertainty_coeff=0.0, uncertainty_decoder_method=None, num_sampled_models=10, num_sampled_outcomes = 40,
                            model_decoding_mode=None, model_decoding_topk_value=None, BO_acquisition_function="UCB", BO_default_value_invalid=0.0,
                            min_bound=-2, max_bound = 2, batch_size=64, generation_timout_seconds=600, model_type="JTVAE"
                            ):
    """
    Bayesian optimization in latent space. Two different modes: BO_uncertainty_mode=="Penalized_objective" (uncertainty-aware surrogate) or BO_uncertainty_mode=="Uncertainty_censoring"
    model_decoding_mode and model_decoding_topk_value are only relevant for RNN decoding (CharVAE).
    """
    smiles_generated_objects = []
    pred_property_values = []

    if model_type=='JTVAE':
        hidden_dim = model.latent_size*2
    elif model_type=='CharVAE':
        hidden_dim = model.params.z_dim
        if model_decoding_mode is not None:
            model.sampling_mode = model_decoding_mode
            model.generation_top_k_sampling = model_decoding_topk_value
    
    #compute actual uncertainty threshold for uncertainty_censoring mode based on percentile
    if BO_uncertainty_mode=="Uncertainty_censoring" and BO_uncertainty_threshold!='No_constraint':
        BO_uncertainty_threshold_value = decoder_uncertainty_stats_training[model_type][uncertainty_decoder_method][BO_uncertainty_threshold]
    
    objects_latent_representation = starting_objects_latent_embeddings.view(-1, hidden_dim)
    objects_properties = starting_objects_properties.view(-1,1)

    for step in tqdm.tqdm(range(number_BO_steps)):
        num_training_points_surrogate = len(objects_latent_representation)
        train_X = objects_latent_representation.detach().to(device)
        train_Y = standardize(objects_properties).detach().to(device)
        
        if BO_uncertainty_mode=="Penalized_objective" and BO_uncertainty_coeff > 0.0:
            with torch.no_grad():
                if step == 0: #On the first step, we compute uncertainty for all starting (latent) points
                    uncertainty_decoder = torch.zeros(num_training_points_surrogate).to(device)
                    for batch_object_indices in range(0, num_training_points_surrogate, batch_size):
                        a, b = batch_object_indices , batch_object_indices+batch_size
                        z_slice = objects_latent_representation[a:b].to(device)
                        uncertainty_decoder[batch_object_indices:batch_object_indices+batch_size] = model.decoder_uncertainty_from_latent(
                                                                                                        z = z_slice,
                                                                                                        method = uncertainty_decoder_method,
                                                                                                        num_sampled_models=num_sampled_models, 
                                                                                                        num_sampled_outcomes=num_sampled_outcomes
                                                                                                        ).squeeze().detach().cpu()
                else:
                    #For all subsequent steps, we just need to compute the uncertainty for the new point and add to previously computed uncertainties
                    new_point_uncertainty_decoder[batch_object_indices:batch_object_indices+batch_size] = model.decoder_uncertainty_from_latent(
                                                                                                        z = generated_object.to(device),
                                                                                                        method = uncertainty_decoder_method,
                                                                                                        num_sampled_models=num_sampled_models, 
                                                                                                        num_sampled_outcomes=num_sampled_outcomes
                                                                                                        )
                    uncertainty_decoder = torch.cat(tensors=(uncertainty_decoder, new_point_uncertainty_decoder.view(1)), dim=0)

                train_Y = train_Y - BO_uncertainty_coeff * standardize(uncertainty_decoder.view(-1,1))
                train_Y = train_Y.detach().to(device)
        
        #Single-task exact GP model
        gp = SingleTaskGP(train_X=train_X, train_Y=train_Y).to(device)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_model(mll)

        #Acquisition function:
        if BO_acquisition_function=="UCB":
            BO_acq_func, q = UpperConfidenceBound(gp, beta=0.1), 1
        elif BO_acquisition_function=="EI":
            BO_acq_func, q = ExpectedImprovement(gp, best_f=0.1), 1
        elif BO_acquisition_function=="PI":
            BO_acq_func, q = ProbabilityOfImprovement(gp, best_f=0.1), 1
        elif BO_acquisition_function=="qUCB":
            BO_acq_func, q = qUpperConfidenceBound(gp, beta=0.1), 20 
        elif BO_acquisition_function=="qEI":
            BO_acq_func, q = qExpectedImprovement(gp, best_f=0.1), 20
        elif BO_acquisition_function=="qNoisyEI":
            BO_acq_func, q = qNoisyExpectedImprovement(gp), 20
        
        #Optimize the acquisition function
        print("Optimizing acq function")
        bounds = torch.stack([torch.ones(hidden_dim) * min_bound, torch.ones(hidden_dim) * max_bound]).to(device)
        generated_object, pred_property_value = optimize_acqf(
                                                            acq_function=BO_acq_func, 
                                                            bounds=bounds, 
                                                            q=q, 
                                                            num_restarts=min(20,num_training_points_surrogate),
                                                            raw_samples=num_training_points_surrogate,
                                                            sequential=True,
                                                            return_best_only=True
                                                            )
        
        generated_object = generated_object.view(-1,hidden_dim)
        with torch.no_grad():
            if BO_uncertainty_mode=="Uncertainty_censoring" and BO_uncertainty_threshold!="No_constraint":
                #Compute uncertainty for each candidate. Check which are below threshold. If at least one, return the one with best predicted value. Otherwise, return lowest uncertainty point.
                uncertainty_generated = torch.zeros(len(generated_object)).to(device)
                for batch_object_indices in range(0, q, batch_size):
                    a, b = batch_object_indices , batch_object_indices+batch_size
                    z_slice = generated_object[a:b].to(device)
                    uncertainty_generated[batch_object_indices:batch_object_indices+batch_size] = model.decoder_uncertainty_from_latent(
                                                                                                            z = z_slice,
                                                                                                            method = uncertainty_decoder_method,
                                                                                                            num_sampled_models=num_sampled_models, 
                                                                                                            num_sampled_outcomes=num_sampled_outcomes
                                                                                                        ).squeeze()
                index_below_uncertainty_threshold = (uncertainty_generated < BO_uncertainty_threshold_value)
                num_below_threshold = index_below_uncertainty_threshold.int().sum()
                if num_below_threshold > 0:
                    generated_object = generated_object[index_below_uncertainty_threshold]
                    pred_property_value = pred_property_value[index_below_uncertainty_threshold]
                    generated_object = generated_object[-1]
                    pred_property_value = pred_property_value[-1]
                else:
                    min_uncertainty_point = uncertainty_generated.argmin()
                    generated_object = generated_object[min_uncertainty_point]
                    pred_property_value = pred_property_value[min_uncertainty_point]
            else:
                if len(generated_object)>1:
                    generated_object = generated_object[-1]
                    pred_property_value = pred_property_value[-1]

        pred_property_values.append(pred_property_value.item())
        generated_object = generated_object.view(1,hidden_dim)
        
        with torch.no_grad():
            if model_type=='JTVAE':
                z = generated_object.view(1,hidden_dim).to(device)
                z_tree, z_mol = z[:,:model.latent_size], z[:,model.latent_size:]
                try:
                    smiles_new_object = func_timeout.func_timeout(generation_timout_seconds, model.decode, args=(z_tree, z_mol), kwargs={'prob_decode':False})
                    new_point_property = compute_target_logP(smiles_new_object, default_value=BO_default_value_invalid)            
                    smiles_generated_objects.append(smiles_new_object)
                    objects_properties = torch.cat(tensors=(objects_properties.float(), torch.tensor(new_point_property).view(1).float()), dim=0)
                    objects_latent_representation = torch.cat(tensors=(objects_latent_representation, generated_object.view(1,hidden_dim)), dim=0)
                except:
                    print("timed out")

            elif model_type=='CharVAE':
                decoded_new_object = model.generate_from_latent(generated_object)
                smiles_new_object = convert_tensors_to_smiles(decoded_new_object, model.params.indices_char)[0]
                smiles_generated_objects.append(smiles_new_object)
                new_point_property = compute_target_logP(smiles_new_object, default_value=BO_default_value_invalid)
                objects_properties = torch.cat(tensors=(objects_properties.float(), torch.tensor(new_point_property).view(1).float()), dim=0)
                objects_latent_representation = torch.cat(tensors=(objects_latent_representation, generated_object.view(1,hidden_dim)), dim=0)
    
    return smiles_generated_objects, pred_property_values