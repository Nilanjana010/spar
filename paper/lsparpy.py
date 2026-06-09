import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sknetwork.hierarchy import tree_sampling_divergence, Paris
from scipy.sparse import csr_matrix

def spar_neuronpedia(json_data, e):

    print("Intial Graph Stats:")

    #Handle 'edges' vs 'links' from JSON
    edge_key = 'edges' if 'edges' in json_data else 'links'

    #Scan keys if standard ones are missing
    if edge_key not in json_data and 'links' not in json_data:
         for k, v in json_data.items():
             if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict) and 'source' in v[0]:
                 edge_key = k
                 break
    
    node_key = 'nodes'
    
    original_nodes = json_data.get(node_key, [])
    original_edges = json_data.get(edge_key, [])
    
    print(f"Detected keys -> Nodes: '{node_key}', Edges: '{edge_key}'")
    print(f"Input Stats: {len(original_nodes)} nodes, {len(original_edges)} edges")

    #Build DAG
    G = nx.DiGraph()
    
    #Add all original nodes first (to ensure ID's tracked correctly)
    node_map = {} # Map ID -> Node Object
    for node in original_nodes:

        #Gather correct neuronpedia tag
        n_id = str(node.get('id') or node.get('node_id') or node.get('name'))
        G.add_node(n_id)
        node_map[n_id] = node

    #Add edges
    for edge in original_edges:
        u = str(edge['source'])
        v = str(edge['target'])
        w = float(edge.get('weight', 0.0))
        
        #Ignore edges pointing to non-existent nodes
        if u in node_map and v in node_map:
            G.add_edge(u, v, weight=w, original_data=edge)

    print("Run L-SPAR")
    
    edges_to_keep = set()

    for v in G.nodes():
        incoming = list(G.in_edges(v, data=True))
        degree = len(incoming)
        
        if degree == 0: continue

        #L-Spar Budget Rule: k = d^e
        k = int(np.floor(degree ** e))

        #k = max(1, int(np.ceil(e * degree)))

        #Sort by Attribution Magnitude
        incoming.sort(key=lambda x: abs(x[2]['weight']), reverse=True)
        
        #L-Spar Budget Rule: k = d^e on the remaining edges
        num_removed_edges = degree - k
        k_new = int(np.floor(num_removed_edges ** e))

        #Keep top k
        try:
            for src, tgt, _ in incoming[:k + k_new]:
                edges_to_keep.add((src, tgt))
        except:
            print("error")        

    print(f"Sparsification complete. Kept {len(edges_to_keep)} edges.")

    print("Check compliance and reconstruct graph")
    
    final_edges = []
    active_node_ids = set()

    #Reconstruct Edges List
    for u, v in edges_to_keep:
        #Retrieve original edge data to preserve extra fields
        original_edge_data = G.edges[u, v]['original_data']
        final_edges.append(original_edge_data)
        
        #Mark nodes as "Active"
        active_node_ids.add(u)
        active_node_ids.add(v)

    #Reconstruct Nodes List (Remove Orphans)
    final_nodes = []
    for n_id, node_obj in node_map.items():
        if n_id in active_node_ids:
            final_nodes.append(node_obj)
            


    #Formatting
    output_json = json_data.copy()
    output_json[edge_key] = final_edges
    output_json[node_key] = final_nodes

    #Update Metadata
    if 'metadata' in output_json:
        output_json['metadata']['node_count'] = len(final_nodes)
        output_json['metadata']['edge_count'] = len(final_edges)
        
        #Append tool info w/ sparsification metrics
        if 'sparsification_info' not in output_json['metadata']:
             output_json['metadata']['sparsification_info'] = {}
        output_json['metadata']['sparsification_info'].update({
            "method": "L-Spar",
            "exponent": e,
            "original_node_count": len(original_nodes),
            "original_edge_count": len(original_edges)
        })

    print(f"Output: {len(final_nodes)} nodes, {len(final_edges)} edges")    
    return output_json

#Save for calculating probabilities
def save_sparsified_graph(input_path, output_path, e):
    with open(input_path, 'r') as f:
        data = json.load(f)
        
    clean_data = spar_neuronpedia(data, e=e)
    
    with open(output_path, 'w') as f:
        json.dump(clean_data, f, indent=2)
    print(f"Saved to {output_path}")



def calculate_graph_probabilities(json_data, temperature=1.0):

    edge_key = 'edges' if 'edges' in json_data else 'links'
    edges = json_data.get(edge_key, [])
    nodes = json_data.get('nodes', [])
    
    #Map node IDs to names for readability
    node_names = {}
    for n in nodes:
        nid = str(n.get('node_id') or n.get('id'))
        #'clerp' has the human-readable labels (e.g., 'Output " rat"')
        label = n.get('clerp') or n.get('name') or nid
        node_names[nid] = label
    
    #Identify potential output predictions
    #Detect all output nodes, not just the single target
    sink_nodes = [
        n.get('node_id') for n in nodes 
        if n.get('is_target_logit') == True or n.get('feature_type') == 'logit'
    ]
    
    print(f"Detected {len(sink_nodes)} output (sink) nodes")
    
    if not sink_nodes:
        print("No output (sink) nodes detected.")
        return {}

    #Aggregate logits for output nodes
    logits = {node_id: 0.0 for node_id in sink_nodes}
    for e in edges:
        t = str(e['target'])
        if t in logits:
            logits[t] += float(e.get('weight', 0.0))
   
    node_ids = list(logits.keys())
    logit_values = np.array([logits[nid] for nid in node_ids]) / temperature

    #Apply Softmax: P_i = exp(w_i / T) / sum(exp(w_j / T))
    """
    e_x = np.exp(logit_values)
    probs = e_x / e_x.sum()
    """
    #Stable Softmax
    shift_logits = logit_values - np.max(logit_values)
    e_x = np.exp(shift_logits)
    probs = e_x / e_x.sum()
    
    #Format results
    results = []
    for i, nid in enumerate(node_ids):
        results.append({
            "node_id": nid,
            "label": node_names.get(nid, "Unknown"),
            "total_logit": round(logit_values[i] * temperature, 4),
            "probability": round(probs[i], 4)
        })
    
    #Sort and return
    return sorted(results, key=lambda x: x['probability'], reverse=True)



def visualize_top_probabilities(results, top_n=10, title="Top Model Outputs"):
    #Slice the top N results
    top_results = results[:top_n]
    
    if not top_results:
        print("No results found.")
        return

    labels = [r['label'] for r in top_results]
    probs = [r['probability'] for r in top_results]
    logits = [r['total_logit'] for r in top_results]

    fig, ax1 = plt.subplots(figsize=(12, 7))

    #Bar chart for probabilities
    color = 'tab:blue'
    bars = ax1.bar(labels, probs, color=color, alpha=0.6, label='Probability')
    ax1.set_ylabel('Probability (0.0 - 1.0)', fontsize=12)
    ax1.tick_params(axis='y')
    ax1.set_ylim(0, max(probs) * 1.1 if probs else 1.0)
    
    #Secondary axis for raw logits
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.plot(labels, logits, color=color, marker='o', markersize=8, linewidth=2, label='Logit')
    ax2.set_ylabel('Summed Logit Score', fontsize=12)
    ax2.tick_params(axis='y')

    plt.title(f"{title} (Top {len(top_results)})", fontsize=15, pad=20)
    ax1.set_xticklabels(labels, rotation=35, ha='right', fontsize=10)
    
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.1%}', ha='center', va='bottom', fontsize=9)

    fig.tight_layout()
    plt.show()


def calculate_ioi_metrics(sparsified_data, correct_label, incorrect_label):
    results = calculate_graph_probabilities(sparsified_data)
    
    l_correct = 0.0
    l_incorrect = 0.0
    
    #Match search (get correct and "incorrect" logits)
    #Measuring faithfulness
    for r in results:
        label_clean = r['label'].lower()
        if correct_label.lower() in label_clean:
            l_correct += r['total_logit']
        elif incorrect_label.lower() in label_clean:
            l_incorrect += r['total_logit']
    
    logit_diff = l_correct - l_incorrect
    return {
        "logit_difference": logit_diff,
        "correct_logit": l_correct,
        "incorrect_logit": l_incorrect
    }

def run_mib_evaluation(original_json_path, exponents, correct_label, incorrect_label):
    with open(original_json_path, 'r') as f:
        original_data = json.load(f)
    
    #Get baseline performance (Original Graph)
    baseline_stats = calculate_ioi_metrics(original_data, correct_label, incorrect_label)
    baseline_diff = baseline_stats['logit_difference']
    
    records = []
    
    #Sweep through sparsity levels
    for e in exponents:
        sparse_data = spar_neuronpedia(original_data, e=e)
        sparse_stats = calculate_ioi_metrics(sparse_data, correct_label, incorrect_label)
        
        #Calculate Faithfulness (Percentage of original logit diff preserved)
        faithfulness = sparse_stats['logit_difference'] / baseline_diff if baseline_diff != 0 else 0
        
        records.append({
            "exponent": e,
            "edges": len(sparse_data.get('edges', sparse_data.get('links', []))),
            "faithfulness": faithfulness,
            "logit_diff": sparse_stats['logit_difference']
        })
    
    return pd.DataFrame(records)

def evaluate_faithfulness(original_results, sparse_results):
    top_token = original_results[0]['label']
    
    #Get its probability in both
    orig_p = original_results[0]['probability']
    sparse_p = next((r['probability'] for r in sparse_results if r['label'] == top_token), 0)
    
    faithfulness = (sparse_p / orig_p) if orig_p > 0 else 0
    print(f"Faithfulness Score for '{top_token}': {faithfulness:.2%}")
    return faithfulness

def calculate_morf_faithfulness_curve(json_data, correct_label, incorrect_label, steps=15):
    """
    Calculates a MoRF curve by iteratively removing the most relevant edges
    and measuring the impact on the IOI logit difference.
    """
    edge_key = 'edges' if 'edges' in json_data else 'links'

    original_edges = json_data.get(edge_key, []).copy()

    #Sort edges by absolute weight (Most Relevant First)
    #This aligns with how your L-Spar function identifies importance
    original_edges.sort(key=lambda x: abs(float(x.get('weight', 0.0))), reverse=True)

    baseline_stats = calculate_ioi_metrics(json_data, correct_label, incorrect_label)
    baseline_diff = baseline_stats['logit_difference']

    results = []
    chunk_size = max(1, len(original_edges) // steps)

    #Deep copy to avoid modifying original data, can drop if I want
    import copy
    current_data = copy.deepcopy(json_data)

    #Range to capture all steps
    for i in range(steps + 1):
        #Update graph state
        start_idx = i * chunk_size
        current_data[edge_key] = original_edges[start_idx:] 
        
        #Calculate the weights of edges revmoed
        removed_so_far = original_edges[:start_idx]
        total_weight_removed = sum(abs(float(e.get('weight', 0.0))) for e in removed_so_far)
        
        #Calculate current performance
        current_stats = calculate_ioi_metrics(current_data, correct_label, incorrect_label)
        
    
        faithfulness = current_stats['logit_difference'] / baseline_diff if baseline_diff != 0 else 0
        
        results.append({
            "edges_removed": min(start_idx, len(original_edges)),
            "edges_remaining": len(current_data[edge_key]),
            "faithfulness": faithfulness,
            "sum_weights_removed": total_weight_removed
        })
    return pd.DataFrame(results)


def calculate_lorf_faithfulness_curve(json_data, correct_label, incorrect_label, steps=15):
    edge_key = 'edges' if 'edges' in json_data else 'links'
    original_edges = json_data.get(edge_key, []).copy()

    # Sort: Least Relevant First (Absolute weight, ascending)
    original_edges.sort(key=lambda x: abs(float(x.get('weight', 0.0))), reverse=False)

    baseline_stats = calculate_ioi_metrics(json_data, correct_label, incorrect_label)
    baseline_diff = baseline_stats['logit_difference']

    results = []
    chunk_size = max(1, len(original_edges) // steps)
    
    #Deep copy to avoid modifying original data, can drop if I want
    import copy
    current_data = copy.deepcopy(json_data)

    #Range to capture all steps
    for i in range(steps + 1):
        #Update graph state
        start_idx = i * chunk_size
        current_data[edge_key] = original_edges[start_idx:] 
        
        #Calculate the weights of edges revmoed
        removed_so_far = original_edges[:start_idx]
        total_weight_removed = sum(abs(float(e.get('weight', 0.0))) for e in removed_so_far)
        
        #Calculate current performance
        current_stats = calculate_ioi_metrics(current_data, correct_label, incorrect_label)
        
    
        faithfulness = current_stats['logit_difference'] / baseline_diff if baseline_diff != 0 else 0
        
        results.append({
            "edges_removed": min(start_idx, len(original_edges)),
            "edges_remaining": len(current_data[edge_key]),
            "faithfulness": faithfulness,
            "sum_weights_removed": total_weight_removed
        })
    return pd.DataFrame(results)

def calc_tsr(json_data, sparsified_data):
    #TSR = (sum of sparsified edge weights) / (sum of original edge weights)
    
    edge_key = 'edges' if 'edges' in json_data else 'links'
    
    original_edges = json_data.get(edge_key, [])
    sparsified_edges = sparsified_data.get(edge_key, [])
    
    #Sum weights in original graph
    original_weight_sum = sum(
        float(e.get('weight', 0.0)) for e in original_edges
    )
    
    #Sum weights in sparsified graph
    sparsified_weight_sum = sum(
        float(e.get('weight', 0.0)) for e in sparsified_edges
    )
    
    tsr = sparsified_weight_sum / original_weight_sum if original_weight_sum != 0 else 0.0
    
    return tsr


def compute_kl_divergence(original_data, sparsified_data, eps=1e-12):
    
    full_results = calculate_graph_probabilities(original_data)
    sparse_results = calculate_graph_probabilities(sparsified_data)
    
    full_probs = {r['label']: r['probability'] for r in full_results}
    sparse_probs = {r['label']: r['probability'] for r in sparse_results}
    
    #Union of tokens
    all_tokens = set(full_probs) | set(sparse_probs)
    
    P = []
    Q = []
    
    for token in all_tokens:
        p = full_probs.get(token, 0.0)
        q = sparse_probs.get(token, 0.0)
        
        #Don't/cant do div. by 0, so have some epsilon.
        P.append(max(p, eps))
        Q.append(max(q, eps))
    
    P = np.array(P)
    Q = np.array(Q)
    
    #Normalize
    P /= P.sum()
    Q /= Q.sum()
    
    kl = np.sum(P * np.log(P / Q))
    
    return kl
    
def compute_tsd(original_data, sparsified_data):
    
    #Compute Tree Sampling Divergence between original and sparsified graphs


    def json_to_adjacency(json_data):
        edge_key = 'edges' if 'edges' in json_data else 'links'
        nodes = json_data.get('nodes', [])
        edges = json_data.get(edge_key, [])

        node_ids = [str(n.get('id') or n.get('node_id') or n.get('name')) for n in nodes]
        node_idx = {nid: i for i, nid in enumerate(node_ids)}
        n = len(node_ids)

        if n == 0:
            return csr_matrix((0, 0))

        rows, cols, data = [], [], []
        for e in edges:
            u = str(e['source'])
            v = str(e['target'])
            w = abs(float(e.get('weight', 0.0)))
            if u in node_idx and v in node_idx and w > 0:
                rows.append(node_idx[u])
                cols.append(node_idx[v])
                data.append(w)
                rows.append(node_idx[v])
                cols.append(node_idx[u])
                data.append(w)

        return csr_matrix((data, (rows, cols)), shape=(n, n))

    try:
        A_orig = json_to_adjacency(original_data)
        A_sparse = json_to_adjacency(sparsified_data)

        if A_orig.shape[0] < 2 or A_sparse.shape[0] < 2:
            return 0.0

        paris = Paris()
        D_orig = paris.fit_transform(A_orig)
        tsd_orig = tree_sampling_divergence(A_orig, D_orig, normalized=True)

        D_sparse = paris.fit_transform(A_sparse)
        tsd_sparse = tree_sampling_divergence(A_sparse, D_sparse, normalized=True)

        return float(tsd_orig - tsd_sparse)
    except Exception as ex:
        print(f"TSD computation error: {ex}")
        return 0.0

#Linear variant of SPAR
def spar_neuronpedia2(json_data, e):

    print("Intial Graph Stats:")

    #Handle 'edges' vs 'links' from JSON
    edge_key = 'edges' if 'edges' in json_data else 'links'

    #Scan keys if standard ones are missing
    if edge_key not in json_data and 'links' not in json_data:
         for k, v in json_data.items():
             if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict) and 'source' in v[0]:
                 edge_key = k
                 break
    
    node_key = 'nodes'
    
    original_nodes = json_data.get(node_key, [])
    original_edges = json_data.get(edge_key, [])
    
    print(f"Detected keys -> Nodes: '{node_key}', Edges: '{edge_key}'")
    print(f"Input Stats: {len(original_nodes)} nodes, {len(original_edges)} edges")

    #Build DAG
    G = nx.DiGraph()
    
    #Add all original nodes first (to ensure ID's tracked correctly)
    node_map = {} # Map ID -> Node Object
    for node in original_nodes:

        #Gather correct neuronpedia tag
        n_id = str(node.get('id') or node.get('node_id') or node.get('name'))
        G.add_node(n_id)
        node_map[n_id] = node

    #Add edges
    for edge in original_edges:
        u = str(edge['source'])
        v = str(edge['target'])
        w = float(edge.get('weight', 0.0))
        
        #Ignore edges pointing to non-existent nodes
        if u in node_map and v in node_map:
            G.add_edge(u, v, weight=w, original_data=edge)

    print("Run L-SPAR")
    
    edges_to_keep = set()

    for v in G.nodes():
        incoming = list(G.in_edges(v, data=True))
        degree = len(incoming)
        
        if degree == 0: continue

        #L-Spar linear Budget Rule: k = d*e
        k = int(np.floor(degree * e))

        #Look at this line (Swati comments) -> this keep a fixed percentage of edges instead of d^e
        #k = max(1, int(np.ceil(e * degree)))

        #Sort by Attribution Magnitude (Absolute Value) -> do we need it to consider negative weights?
        incoming.sort(key=lambda x: abs(x[2]['weight']), reverse=True)
        
        #L-Spar linear Budget Rule: k = d*e on the remaining edges
        num_removed_edges = degree - k
        k_new = int(np.floor(num_removed_edges * e))

        #Keep top k
        try:
            for src, tgt, _ in incoming[:k + k_new]:
                edges_to_keep.add((src, tgt))
        except:
            print("error")        

    print(f"Sparsification complete. Kept {len(edges_to_keep)} edges.")

    print("Check compliance and reconstruct graph")
    
    final_edges = []
    active_node_ids = set()

    #Reconstruct Edges List
    for u, v in edges_to_keep:
        #Retrieve original edge data to preserve extra fields
        original_edge_data = G.edges[u, v]['original_data']
        final_edges.append(original_edge_data)
        
        #Mark nodes as "Active"
        active_node_ids.add(u)
        active_node_ids.add(v)

    #Reconstruct Nodes List (Remove Orphans)
    final_nodes = []
    for n_id, node_obj in node_map.items():
        if n_id in active_node_ids:
            final_nodes.append(node_obj)
            


    #Formatting
    output_json = json_data.copy()
    output_json[edge_key] = final_edges
    output_json[node_key] = final_nodes

    #Update Metadata
    if 'metadata' in output_json:
        output_json['metadata']['node_count'] = len(final_nodes)
        output_json['metadata']['edge_count'] = len(final_edges)
        
        #Append tool info w/ sparsification metrics
        if 'sparsification_info' not in output_json['metadata']:
             output_json['metadata']['sparsification_info'] = {}
        output_json['metadata']['sparsification_info'].update({
            "method": "L-Spar",
            "exponent": e,
            "original_node_count": len(original_nodes),
            "original_edge_count": len(original_edges)
        })

    print(f"Output: {len(final_nodes)} nodes, {len(final_edges)} edges")    
    return output_json
