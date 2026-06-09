import os
import json
import sys
import numpy as np
import networkx as nx
import contextlib
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import lsparpy
from math import inf

class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

#Define Directories
INPUT_DIR = "in_graph_files_adv_gemma"
PLOT_DIR = "plots_adv_gemma"
SPARSE_DIR = "sparsified_adv_gemma"

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(SPARSE_DIR, exist_ok=True)

global_records = []

def getcorrectlabel(json_data):
    nodes = json_data.get('nodes', [])
    for n in nodes:
        #Check for both Boolean True and String "true"
        if n.get('is_target_logit') in [True, 'true'] and n.get('feature_type') == 'logit':
            parts = n.get('clerp', '').split('\"')
            return parts[1] if len(parts) > 1 else ""
    return ""

#Chose second most likely as incorr.
def getincorrectlabel(json_data):
    nodes = json_data.get('nodes', [])

    #Sort all logit nodes by probability descending
    logit_nodes = [n for n in nodes if n.get('feature_type') == 'logit']
    logit_nodes.sort(key=lambda x: x.get('token_prob', 0), reverse=True)
    #[0] is the correct label (highest prob)
    #[1] is the second most likely label (incorr)
    if len(logit_nodes) > 1:
        second_best = logit_nodes[1]
        clerp = second_best.get('clerp').split('\"')
        incorrect_label = clerp[1]
    else:
        incorrect_label = ""

    print("incorrect label (2nd place) = ", incorrect_label)
    return incorrect_label

#Exponents to get metrics / sparsification
exponents = [0.9, 0.95, .97, .99, 1]

#Process Directory Structure
for subdir in os.listdir(INPUT_DIR):
    sub_dir_path = os.path.join(INPUT_DIR, subdir)
    if not os.path.isdir(sub_dir_path):
        continue
        
    print(f"\n--- Processing Directory: {subdir} ---")
    
    for ind_file in os.listdir(sub_dir_path):
        if "graph-metadata" in ind_file or not ind_file.endswith(".json"):
            continue
            
        print("Processing:", ind_file)
        file_path = os.path.join(sub_dir_path, ind_file)
        base_name = os.path.splitext(ind_file)[0]

        #Setup logging and plot folders
        plot_folder = os.path.join(PLOT_DIR, base_name + "_plots")
        os.makedirs(plot_folder, exist_ok=True)
        log_file_path = os.path.join(plot_folder, "adv_evaluation_results.log")
                
        with open(log_file_path, 'w') as log_f:
            with contextlib.redirect_stdout(Tee(sys.stdout, log_f)):
                print(f"Target File: {ind_file}")
                
                #Load original data and labels
                with open(file_path, 'r') as f:
                    original_json = json.load(f)
                
                correct_label = getcorrectlabel(original_json)
                incorrect_label = getincorrectlabel(original_json)

                #Baseline metrics
                full_probs = lsparpy.calculate_graph_probabilities(original_json)
                #Run MIB Evaluation for the curve
                df_mib = lsparpy.run_mib_evaluation(
                    file_path,
                    exponents,
                    correct_label,
                    incorrect_label
                )

                #Plot Faithfulness Curve
                plt.figure(figsize=(8, 5))
                plt.plot(df_mib['edges'], df_mib['faithfulness'], marker='o', color='crimson')
                plt.xlabel("Edges in Circuit")
                plt.ylabel("Faithfulness")
                plt.title(f"Adversarial Faithfulness vs Sparsity: {base_name}")
                plt.grid(True)
                plt.savefig(os.path.join(plot_folder, "MIB_curve_adv.png"))
                plt.close()

                #Iterate through exponents to get metrics
                for e in exponents:
                    print(f"\n--- Sparsity exponent: {e}")
                    sparse_path = os.path.join(SPARSE_DIR, f"{base_name}_e{e}.json")

                    lsparpy.save_sparsified_graph(file_path, sparse_path, e=e)
                    with open(sparse_path, 'r') as f:
                        sparse_json = json.load(f)

                    sparse_probs = lsparpy.calculate_graph_probabilities(sparse_json)
                    faithfulness = lsparpy.evaluate_faithfulness(full_probs, sparse_probs)
                    
                    sparse_ioi = lsparpy.calculate_ioi_metrics(sparse_json, correct_label, incorrect_label)
                    logit_diff = sparse_ioi['logit_difference']

                    tsr = lsparpy.calc_tsr(original_json, sparse_json)
                    kl = lsparpy.compute_kl_divergence(original_json, sparse_json)

                    #Append to global records
                    global_records.append({
                        "graph": base_name,
                        "exponent": e,
                        "edges": len(sparse_json.get('edges', sparse_json.get('links', []))),
                        "logit_diff": logit_diff,
                        "faithfulness": faithfulness,
                        "tsr": tsr,
                        "kl_divergence": kl
                    })
                    print(f"TSR: {tsr:.4f} | KL: {kl:.4f} | Faithfulness: {faithfulness:.4f} | Logit diff: {logit_diff:.4f}")

                #Calculate MoRF/LoRF Curves
                full_morf = lsparpy.calculate_morf_faithfulness_curve(original_json, correct_label, incorrect_label)
                sparse_morf = lsparpy.calculate_morf_faithfulness_curve(sparse_json, correct_label, incorrect_label)
                full_lorf = lsparpy.calculate_lorf_faithfulness_curve(original_json, correct_label, incorrect_label)
                sparse_lorf = lsparpy.calculate_lorf_faithfulness_curve(sparse_json, correct_label, incorrect_label)

                #Save tables
                full_morf.to_csv(os.path.join(plot_folder, "morf_full.csv"), index=False)
                sparse_morf.to_csv(os.path.join(plot_folder, "morf_sparse.csv"), index=False)
                full_lorf.to_csv(os.path.join(plot_folder, "lorf_full.csv"), index=False)
                sparse_lorf.to_csv(os.path.join(plot_folder, "lorf_sparse.csv"), index=False)

                print("Saved MoRF/LoRF tables.")
                print(f"--- Finished processing {ind_file} ---")

#Save Aggregated Global Metrics
global_df = pd.DataFrame(global_records)
global_metrics_path = os.path.join(PLOT_DIR, "global_metrics_adv_gemma.csv")
global_df.to_csv(global_metrics_path, index=False)

print(f"Adversarial global metrics saved to: {global_metrics_path}")
