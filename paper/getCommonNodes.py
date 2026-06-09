import os
import json
import csv
from collections import Counter

def analyze_to_csv(directory, min_freq_pct=0.6, influence_threshold=0.01):
    node_counts = Counter()
    node_metadata = {}
    predictions = []
    
    files = [f for f in os.listdir(directory) if f.endswith('.json')]
    if not files:
        print("No JSON files found in the directory.")
        return
    
    total_files = len(files)
    occurrence_threshold = total_files * min_freq_pct

    for filename in files:
        filepath = os.path.join(directory, filename)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            prompt_tokens = data.get('metadata', {}).get('prompt_tokens', [])
            #Capture Next Token Prediction (Target Logit)
            target_node = next((n for n in data.get('nodes', []) if n.get('is_target_logit')), None)
            pred_token = target_node.get('clerp', "Unknown") if target_node else "N/A"
            predictions.append({'filename': filename, 'predicted_token': pred_token})

            #Filter nodes by attribution (influence) and track frequency
            current_file_nodes = set()
            for node in data.get('nodes', []):
                influence = node.get('influence')
                if influence is not None and influence >= influence_threshold:
                    nid = node['node_id']
                    current_file_nodes.add(nid)
                    
                    if nid not in node_metadata:
                        ctx_idx = node.get('ctx_idx')
                        token = prompt_tokens[ctx_idx] if ctx_idx is not None and ctx_idx < len(prompt_tokens) else "N/A"
                        node_metadata[nid] = {
                            'layer': node.get('layer'),
                            'token_context': token,
                            'feature_type': node.get('feature_type', 'N/A')
                        }
                        node_counts.update(current_file_nodes)
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    #Write Common Nodes to CSV
    common_nodes_file = 'common_nodes.csv'
    with open(common_nodes_file, 'w', newline='') as csvfile:
        fieldnames = ['node_id', 'frequency_count', 'frequency_pct', 'layer', 'token_context', 'feature_type']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for nid, count in node_counts.items():
            if count >= occurrence_threshold:
                meta = node_metadata.get(nid, {'layer': 'N/A', 'token_context': 'N/A', 'feature_type': 'N/A'})
                writer.writerow({
                    'node_id': nid,
                    'frequency_count': count,
                    'frequency_pct': f"{(count / total_files):.2%}",
                    'layer': meta['layer'],
                    'token_context': meta['token_context'],
                    'feature_type': meta['feature_type']
                })

    #Write Predictions to CSV
    predictions_file = 'next_token_predictions.csv'
    with open(predictions_file, 'w', newline='') as csvfile:
        fieldnames = ['filename', 'predicted_token']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)

    print(f"Analysis complete. Found {len([n for n, c in node_counts.items() if c >= occurrence_threshold])} nodes.")

#Run
target_directory = "/path/to/sparsified/graphs"
analyze_to_csv(target_directory, min_freq_pct=0.6, influence_threshold=0.01)
