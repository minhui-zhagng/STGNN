"""Historical graph pipeline, portable paths and explicit edge-mode switch.
Each graph reads its own CSV. Historical label-derived temporal edges remain.
Not a verified causal implementation. Historical temporal mode is explicitly authorized.
"""
from pathlib import Path
from common import load_config,arguments
CFG={};OUT=None;MODE=None
import os

import csv

import itertools

from torch_geometric.data import Data

from scipy.stats import kurtosis, skew

from scipy.fft import fft

import numpy as np

import torch

def extract_features( data_L):
    def extract_one(signal):
        signal = np.array(signal)
        N = len(signal)
        fft_vals = np.abs(fft(signal))
        freqs = fft_vals[:N // 2]  

        top3 = np.sort(freqs)[-3:]


        energy = np.sum(freqs ** 2)


        freq_indices = np.arange(len(freqs))
        center_freq = np.sum(freq_indices * freqs) / (np.sum(freqs) + 1e-8)


        p = freqs / (np.sum(freqs) + 1e-8)
        spectrum_entropy = -np.sum(p * np.log(p + 1e-8))

        return [
            np.mean(signal), np.std(signal), np.max(signal), np.min(signal),
            np.sqrt(np.mean(np.square(signal))),        
            kurtosis(signal),                            
            skew(signal),                              
            np.ptp(signal),                             
            top3[-1], top3[-2], top3[-3],                
            energy, center_freq, spectrum_entropy        
        ]


    feat_L = extract_one(data_L)

    features = np.array(feat_L) 

    # Standardization
    features = (features - features.mean()) / (features.std() + 1e-8)

    return torch.tensor(features, dtype=torch.float32)

def csv_read(CSV_data, CSV_number, file_idx):

    data_L = []

    CSV_total = [
        [123, 161, 158, 122, 52],
        [491, 161, 533, 42, 339],
        [2538, 2496, 371, 1515, 114]
    ]
    CSV_path = ["35Hz12kN", "37.5Hz11kN", "40Hz10kN"]

    base_dir = CFG["data_root"]
    bearing_path = os.path.join(base_dir, CSV_path[CSV_data - 1], f"Bearing{CSV_data}_{CSV_number}")

    num_files = CFG["file_counts"][f"{CSV_data}_{CSV_number}"]

    if not 1 <= file_idx <= num_files:
        raise ValueError(f'CSV index out of range: {file_idx}')
    file_path = os.path.join(bearing_path, f"{file_idx}.csv")
    if not os.path.isfile(file_path):
        raise FileNotFoundError(file_path)
    with open(file_path, "r", newline='') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) < 2:raise ValueError(f'Malformed row in {file_path}')
            data_L.append(float(row[0]))
    if not data_L or not np.isfinite(data_L).all():
        raise ValueError(f'Empty/nonfinite signal: {file_path}')

    return data_L

def get_custom_edge_index():
    edges = [
        (0, 1), (0, 2), (0, 3), (0, 4),
        (1, 4), (2, 4), (3, 4),
        (1, 5), (2, 5), (3, 5), (4, 5),
    ]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index

def compute_markov_transition_prob(label_dir, train_targets, num_states=6):
    transition_counts = np.zeros((num_states, num_states), dtype=int)

    for csv_data, csv_number in train_targets:
        label_path = os.path.join(label_dir, f"label_{csv_data}_{csv_number}.npy")
        if not os.path.exists(label_path):
            continue
        labels = np.load(label_path)
        for i in range(len(labels) - 1):
            src = labels[i]
            dst = labels[i + 1]
            if 0 <= src < num_states and 0 <= dst < num_states:
                transition_counts[src][dst] += 1

    transition_probs = transition_counts.astype(float)
    for i in range(num_states):
        row_sum = transition_probs[i].sum()
        if row_sum > 0:
            transition_probs[i] /= row_sum

    return transition_probs

def generate_edge_attr(edge_index, file_idx, labels, transition_probs, alpha=0):
    mode=MODE
    if mode=='no_edge':return torch.ones((edge_index.shape[1],1),dtype=torch.float32)
    result=[]
    for src,dst in edge_index.t().tolist():
        prior=float(transition_probs[src,dst]);temporal=0.
        if mode!='markov':
            a=np.flatnonzero(np.asarray(labels)==src);b=np.flatnonzero(np.asarray(labels)==dst)
            if len(a) and len(b) and file_idx>=a[0]:
                if file_idx>b[0]:temporal=1.
                else:
                    progress=float(np.clip((file_idx-a[0])/(b[0]-a[0]+1e-5),0,1))
                    temporal=progress if mode in ['linear','linear_markov'] else float(np.expm1(CFG['beta']*progress)/np.expm1(CFG['beta']))
        value=prior if mode=='markov' else temporal
        if mode.endswith('_markov'):value=CFG['alpha']*prior+(1-CFG['alpha'])*temporal
        result.append([value])
    return torch.tensor(result,dtype=torch.float32)


def process_one_sample(csv_data, csv_number, file_idx, label, status_embedding, transition_probs):
    # vibration data reading and feature extraction
    data_L = csv_read(csv_data, csv_number, file_idx)
    feat = extract_features( data_L)

    #node
    node_features = []
    for i in range(6):
        id_code = status_embedding(torch.tensor(i))
        full_feat = torch.cat([feat, id_code], dim=0)
        node_features.append(full_feat)
    x = torch.stack(node_features)

    edge_index = get_custom_edge_index()

    # read label
    label_path = os.path.join(
        str(OUT / "labels"),
        f"label_{csv_data}_{csv_number}.npy"
    )
    labels_all = np.load(label_path)

   
    edge_attr = generate_edge_attr(edge_index, file_idx, labels_all, transition_probs, alpha=CFG["alpha"])

    y = torch.tensor(label, dtype=torch.long)
    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    graph.idx = file_idx
    graph.bearing = f'{csv_data}_{csv_number}'
    return graph

def build_dataset(targets, transition_probs):
    dataset = []
    base_dir = str(OUT / "labels")

    for csv_data, csv_number in targets:
        label_path = os.path.join(base_dir, f"label_{csv_data}_{csv_number}.npy")
        if not os.path.exists(label_path):
            raise FileNotFoundError(label_path)
        labels = np.load(label_path)
        for i, label in enumerate(labels):
            g = process_one_sample(csv_data, csv_number, i + 1, label, status_embedding, transition_probs)
            dataset.append(g)
    return dataset

def main():
    global CFG,OUT,MODE,status_embedding
    args=arguments('Build historical graph data for one edge mode')
    CFG,OUT=load_config(args.config);MODE=args.mode
    if MODE not in CFG['edge_modes']:raise ValueError('Unknown edge mode')
    torch.manual_seed(CFG['seed']);status_embedding=torch.nn.Embedding(6,6)
    folder=OUT/'graphs'/MODE;folder.mkdir(parents=True,exist_ok=True)
    prior=compute_markov_transition_prob(str(OUT/'labels'),CFG['train_bearings'])
    np.save(folder/'markov.npy',prior)
    for part in ['train','test']:
        graphs=build_dataset(CFG[part+'_bearings'],prior)
        expected=sum(CFG['file_counts'][f'{c}_{b}'] for c,b in CFG[part+'_bearings'])
        if len(graphs)!=expected:raise RuntimeError(f'{part}: expected{expected}, got{len(graphs)}; inspect missing/failed files')
        torch.save(graphs,folder/(part+'.pt'))
    import json
    (folder/'config.json').write_text(json.dumps(CFG,indent=2),encoding='utf-8')

if __name__=='__main__':main()
