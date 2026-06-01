from circuit_tracer import ReplacementModel, attribute
from circuit_tracer.utils import create_graph_files
from datasets import load_dataset
from pathlib import Path
import torch, gc, os
from ordered_set import OrderedSet


def cleanup(model_obj):
    print("Performing manual cleanup...")
    try:
        # Manually reset hooks to prevent the TypeError in __del__
        if hasattr(model_obj, 'model'):
            model_obj.model.reset_hooks()
    except:
        pass
    del model_obj
    gc.collect()
    torch.cuda.empty_cache()

ds = load_dataset("walledai/AdvBench")

model_name = "meta-llama/llama-3.2-1B"
transcoder_name = "mntss/clt-llama-3.2-1b-524k"

backend = 'transformerlens'
model = ReplacementModel.from_pretrained(
    model_name, transcoder_name, dtype=torch.bfloat16, backend=backend, device_map="auto"
)

unique_prompts = []

for ds_load in range(len(ds['train'])):
     unique_prompts.append(ds['train'][ds_load]['prompt'])
   
for indx, prompt in enumerate(unique_prompts):
                
        max_n_logits = 30  # How many logits to attribute from, max. We attribute to min(max_n_logits, n_logits_to_reach_de>
        desired_logit_prob = 0.95  # Attribution will attribute from the minimum number of logits needed to reach this prob>
        max_feature_nodes = 8192  # Only attribute from this number of feature nodes, max. Lower is faster, but you will lo>
        batch_size = 64  # Batch size when attributing
        verbose = True  # Whether to display a tqdm progress bar and timing report
        
        try:
            graph = attribute(
                prompt=prompt,
                model=model,
                max_n_logits=max_n_logits,
                desired_logit_prob=desired_logit_prob,
                batch_size=batch_size,
                max_feature_nodes=max_feature_nodes,
                offload=None,
                verbose=verbose,
            )
        
            graph_dir = "graphs_llama_advbench"
            graph_name = "prompt_" + str(indx) + ".pt"
            graph_dir = Path(graph_dir)
            graph_dir.mkdir(exist_ok=True)
            graph_path = graph_dir / graph_name

            graph.to_pt(graph_path)

            os.makedirs("in_graph_files_llama_advbench", exist_ok=True)
            nm = "prompt_" + str(indx)
            slug =  nm + "_slug" #this is the name that you assign to the graph
            graph_file_nm = nm + "_graph_file"
            graph_file_dir = "./in_graph_files_llama_advbench/" + graph_file_nm 
            node_threshold = 0.7  #keep only the minimum # of nodes whose cumulative influence is >= 0.8
            edge_threshold = 0.95  #keep only the minimum # of edges whose cumulative influence is >= 0.98

            create_graph_files(
                graph_or_path=graph_path,  # the graph to create files for
                slug=slug,
                output_path=graph_file_dir,
                node_threshold=node_threshold,
                edge_threshold=edge_threshold)

        except Exception as e:
            print(f"Error on prompt {indx}: {e}")
        finally:
            torch.cuda.empty_cache()  
cleanup(model)