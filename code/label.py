"""Historical labeling: known fault-mode map and97% failure boundary.
This is NOT a verified8Ah instantaneous failure implementation.
"""
from common import load_config,arguments
CFG={}
#this code is to decide the start point of single fault and multiple, then label each csv file
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.fft import fft

# ========= parameters=========
fs = 25600
window_size = 20480
step = 1
sigma_scale = 3
min_count = 3


n_ball = 8
ball_d = 7.92
pitch_d = 34.55
cos_phi = 1  # 接触角 0°
fr_dict = {1: 35.0, 2: 37.5, 3: 40.0}

# ========= characteristic frequency calculation =========
def calc_freqs(fr):
    bpfo = (n_ball / 2) * fr * (1 - ball_d / pitch_d)
    bpfi = (n_ball / 2) * fr * (1 + ball_d / pitch_d)
    ftf  = 0.5 * fr * (1 - ball_d / pitch_d)
    return {'BPFO': bpfo, 'BPFI': bpfi, 'FTF': ftf}

# ========= fault start identification =========
def detect_start(energy, ratio=0.3, N=3, K=3):
    base_len = int(len(energy) * ratio)
    mu = np.mean(energy[:base_len])
    sigma = np.std(energy[:base_len])
    threshold = mu + N * sigma
    for i in range(len(energy) - K + 1):
        if all(e > threshold for e in energy[i:i+K]):
            return i, threshold
    return None, threshold

# ========= label =========

fault_label_map = {
    (1, 1): 2,   
    (1, 2): 2,  
    (1, 3): 2,  
    (1, 4): 3,   
    (1, 5): 4, 
    (2, 1): 1,   
    (2, 2): 2,   
    (2, 3): 3,   
    (2, 4): 2,  
    (2, 5): 2,   
    (3, 1): 2,   
    (3, 2): 4, 
    (3, 3): 1,  
    (3, 4): 1,   
    (3, 5): 2,   
}

fault_type_names = {
    0: "Normal",
    1: "Inner Race Fault",
    2: "Outer Race Fault",
    3: "Cage Fault",
    4: "Multiple Faults",
    5: "Failure"
}
def label_bearing(CSV_data, CSV_number, fault_label, save_path=None):
    fr = fr_dict[CSV_data]
    freqs = calc_freqs(fr)
    freq_map = {1: 'BPFI', 2: 'BPFO', 3: 'FTF', 4: ['BPFI', 'BPFO']}
    fault_freqs = freq_map[fault_label]

    CSV_total = [[123, 161, 158, 122, 52], [491, 161, 533, 42, 339], [2538, 2496, 371, 1515, 114]]
    CSV_path = ["35Hz12kN", "37.5Hz11kN", "40Hz10kN"]
    base_dir = CFG["data_root"]
    bearing_path = os.path.join(base_dir, CSV_path[CSV_data - 1], f"Bearing{CSV_data}_{CSV_number}")
    num_files = CSV_total[CSV_data - 1][CSV_number - 1]

    # ---extract energy ---
    energies = []
    for i in range(1, num_files + 1, step):
        file_path = os.path.join(bearing_path, f"{i}.csv")
        if not os.path.exists(file_path): continue
        with open(file_path, "r") as f:
            reader = csv.reader(f)
            next(reader)
            sig = [float(row[CFG["labeling"]["channel"]]) for row in reader if len(row) > 1]
        if len(sig) < window_size: continue
        seg = sig[:window_size]
        env = np.abs(hilbert(seg))
        spec = np.abs(fft(env))[:len(env)//2]
        freqs_fft = np.linspace(0, fs / 2, len(env)//2)

        # multiple fault max(BPFO, BPFI)
        if isinstance(fault_freqs, list):
            energy = max(np.mean(spec[(freqs_fft >= freqs[f] - 2) & (freqs_fft <= freqs[f] + 2)]) for f in fault_freqs)
        else:
            f0 = freqs[fault_freqs]
            mask = (freqs_fft >= f0 - 2) & (freqs_fft <= f0 + 2)
            energy = np.mean(spec[mask])
        energies.append(energy)

    L = len(energies)
    time_steps = list(range(L))

    # --- start point definition ---
    fault_start, threshold = detect_start(energies, ratio=CFG["labeling"]["baseline_ratio"], N=CFG["labeling"]["N"], K=CFG["labeling"]["K"])
    failure_start = int(L * CFG["labeling"].get("failure_ratio", 0.97))   #change based on the results of complete_fault.py

    # --- generate label ---
    labels = np.zeros(L, dtype=int)
    if fault_start is not None:
        labels[fault_start:failure_start] = fault_label
    labels[failure_start:] = 5  #complete fault

    # --- visualization---
    plt.figure(figsize=(10, 4))
    plt.plot(time_steps, energies, label="Band Energy", marker='o')
    plt.axhline(threshold, color='orange', linestyle='--', label="Detection Threshold")
    if fault_start is not None:
        plt.axvline(fault_start, color='red', linestyle='--', label=f"Fault Start: {fault_start}")
    plt.axvline(failure_start, color='black', linestyle='--', label=f"Failure Start: {failure_start}")
    plt.title(f"Bearing{CSV_data}_{CSV_number} - Label: {fault_label}")
    plt.xlabel("Time Step")
    plt.ylabel("Band Energy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.close()


    plt.figure(figsize=(10, 2.5))
    plt.step(time_steps, labels, where='post', color='purple')
    plt.title("Label Evolution Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Label")
    plt.yticks([0, 1, 2, 3, 4, 5], ['Normal', 'Inner', 'Outer', 'Cage', 'Multi', 'Failure'])
    plt.grid(True)
    plt.tight_layout()
    plt.close()
    # --- save---
    if save_path:
        np.save(os.path.join(save_path, f"label_{CSV_data}_{CSV_number}.npy"), labels)
        print(f"✅ label have been saved to：{save_path}")
    print(
        f"🔍 sample：Bearing{CSV_data}_{CSV_number}，type：{fault_type_names[fault_label]} (label {fault_label})")
    return labels, fault_start, failure_start



def main():
    global CFG,window_size
    args=arguments('Generate historical labels; see documented limitations')
    CFG,out=load_config(args.config);window_size=CFG['labeling']['window_size']
    folder=out/'labels';folder.mkdir(parents=True,exist_ok=True)
    for c,b in CFG['train_bearings']+CFG['test_bearings']:
        label_bearing(c,b,fault_label_map[(c,b)],str(folder))
    import json
    (folder/'labeling_config.json').write_text(json.dumps(CFG['labeling'],indent=2),encoding='utf-8')

if __name__=='__main__':main()
