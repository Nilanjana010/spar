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

INPUT_DIR = "in_graph_files_gemma_advbench"
PLOT_DIR = "plots_gemma_advbench"
SPARSE_DIR = "sparsified_gemma_advbench"


os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(SPARSE_DIR, exist_ok=True)

global_records = []

def getcorrectlabel(json_data):
    nodes = json_data.get('nodes', [])
    for n in nodes:
        if n.get('is_target_logit') in [True, 'true'] and n.get('feature_type') == 'logit':
            parts = n.get('clerp', '').split('\"')
            return parts[1] if len(parts) > 1 else ""
    return ""

def getincorrectlabel(json_data):
    nodes = json_data.get('nodes', [])
    
    logit_nodes = [n for n in nodes if n.get('feature_type') == 'logit']
    logit_nodes.sort(key=lambda x: x.get('token_prob', 0), reverse=True)
    
    if len(logit_nodes) > 1:
        second_best = logit_nodes[1]
        clerp = second_best.get('clerp').split('\"')
        incorrect_label = clerp[1]
    else:
        incorrect_label = ""
        print("incorrect label (2nd place) = ", incorrect_label)
    return incorrect_label

exponents = [.8, .9, .95, .97, 1]

for subdir  in os.listdir(INPUT_DIR):
    print(f"\n--- Processing Directory: {subdir} ---")
    sub_dir = os.path.join(INPUT_DIR, subdir)
    
    for ind_file in os.listdir(sub_dir):
        if "graph-metadata" in ind_file:
            continue
        print("Processing:", ind_file)
        file_path = os.path.join(sub_dir, ind_file)
        base_name = os.path.splitext(ind_file)[0]


        plot_folder = os.path.join(PLOT_DIR, base_name + "_plots")
        os.makedirs(plot_folder, exist_ok=True)
                
        log_file_path = os.path.join(plot_folder, "evaluation_results.log")
        with open(log_file_path, 'w') as f:
                with contextlib.redirect_stdout(Tee(sys.stdout, f)):
                        print(f"Target File: {ind_file}")
                        output_json = os.path.join("sparsified_advbench_gemma2_paraphrase", ind_file + "_sparsified" + '.json')

                        original_json = json.load(open(file_path))
                        correct_label = getcorrectlabel(original_json)
                        incorrect_label = getincorrectlabel(original_json)

                        full_probs = lsparpy.calculate_graph_probabilities(original_json)
                        baseline_ioi = lsparpy.calculate_ioi_metrics(
                                        original_json,
                                        correct_label,
                                        incorrect_label
                                        )

                        baseline_logit_diff = baseline_ioi['logit_difference']

                        df = lsparpy.run_mib_evaluation(
                            file_path,
                        exponents,
                        correct_label,
                        incorrect_label
                        )

                        plt.figure(figsize=(8, 5))
                        plt.plot(df['edges'], df['faithfulness'], marker='o')
                        plt.xlabel("Edges in Circuit")
                        plt.ylabel("Faithfulness")
                        plt.title("MIB Faithfulness vs Sparsity")
                        plt.grid(True)

                        plt.savefig(os.path.join(plot_folder, "MIB_curve.png"))
                        plt.close()

                        for e in exponents:

                                print("\n--- Sparsity exponent:", e)

                                sparse_path = os.path.join(
                                        SPARSE_DIR,
                                        f"{base_name}_e{e}.json"
                                )

                                lsparpy.save_sparsified_graph(
                                     file_path,
                                        sparse_path,
                                        e=e
                                )

                                sparse_json = json.load(open(sparse_path))

                                
                                #Graph probabilities
                                sparse_probs = lsparpy.calculate_graph_probabilities(
                                        sparse_json
                                )

                                #Faithfulness

                                faithfulness = lsparpy.evaluate_faithfulness(
                                        full_probs,
                                        sparse_probs
                                )

                                

                                sparse_ioi = lsparpy.calculate_ioi_metrics(
                                        sparse_json,
                                        correct_label,
                                        incorrect_label
                                )

                                logit_diff = sparse_ioi['logit_difference']

                                #TSR+KL

                                tsr = lsparpy.calc_tsr(
                                        original_json,
                                        sparse_json
                                )

                                kl = lsparpy.compute_kl_divergence(
                                        original_json,
                                        sparse_json)

                                #Add global metrics

                                global_records.append({
                                        "graph": base_name,
                                        "exponent": e,
                                        "edges": len(
                                        sparse_json.get(
                                                'edges',
                                                sparse_json.get('links', [])
                                        )
                                        ),
                                        "logit_diff": logit_diff,
                                        "faithfulness": faithfulness,
                                        "tsr": tsr,
                                        "kl_divergence": kl
                                })

                                print(
                                        f"TSR: {tsr:.4f} | "
                                        f"KL: {kl:.4f} | "
                                        f"Faithfulness: {faithfulness:.4f} | "
                                        f"Logit diff: {logit_diff:.4f}"
                                )

                full_morf = lsparpy.calculate_morf_faithfulness_curve(
                    original_json,
                    correct_label,
                    incorrect_label)

                sparse_morf = lsparpy.calculate_morf_faithfulness_curve(
                    sparse_json,
                    correct_label,
                    incorrect_label)

                full_lorf = lsparpy.calculate_lorf_faithfulness_curve(
                    original_json,
                    correct_label,
                    incorrect_label)
                sparse_lorf = lsparpy.calculate_lorf_faithfulness_curve(
                    sparse_json,
                    correct_label,
                    incorrect_label)

                #Save tables

                full_morf.to_csv(
                    os.path.join(plot_folder, "morf_full.csv"),
                    index=False)

                sparse_morf.to_csv(
                    os.path.join(plot_folder, "morf_sparse.csv"),
                    index=False)

                full_lorf.to_csv(
                    os.path.join(plot_folder, "lorf_full.csv"),
                    index=False)

                sparse_lorf.to_csv(
                    os.path.join(plot_folder, "lorf_sparse.csv"),
                    index=False)

                print("Saved MoRF/LoRF tables.")
                print("\n--- Finished processing", ind_file, "---")

#Global metrics
global_df = pd.DataFrame(global_records)

global_metrics_path = os.path.join(
    PLOT_DIR,
    "global_metrics_advbench_gemma.csv"
)

global_df.to_csv(global_metrics_path, index=False)

print("Global metrics saved to:")
print(global_metrics_path)
