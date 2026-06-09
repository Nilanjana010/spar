from circuit_tracer.graph import prune_graph
from circuit_tracer import ReplacementModel, attribute
from circuit_tracer.utils import create_graph_files
from datasets import load_dataset
from pathlib import Path
import torch, gc, os

#Change based on new datasets
ds = load_dataset("izi-ano/CounselBench-Adv")
print("ds = ", ds, type(ds))

def cleanup(model_obj):
    print("Performing manual cleanup...")
    try:
        #Manually reset hooks to prevent the TypeError in __del__
        if hasattr(model_obj, 'model'):
            model_obj.model.reset_hooks()
    except:
        pass
    del model_obj
    gc.collect()
    torch.cuda.empty_cache()

#Once you run the gemma 2b model, make sure to comment these two lines out and uncomment the meta llama lines
model_name = "google/gemma-2-2b"
#Pulled from HF: https://huggingface.co/mntss/clt-gemma-2-2b-2.5M
transcoder_name = "mntss/clt-gemma-2-2b-426k"


#Todo: Uncomment and run these on alt CLT!
#model_name = "meta-llama/llama-3.2-1B"
#transcoder_name = "mntss/clt-llama-3.2-1b-524k"

backend = 'transformerlens'
model = ReplacementModel.from_pretrained(
    model_name, transcoder_name, dtype=torch.bfloat16, backend=backend, device_map="auto"
)

#May need to change based on new dataset splits
for category in ds['train'][0].keys():
    print("category = ", category)
    for prompt_num in range(len(ds['train'])):
        
        prompt = ds['train'][prompt_num][category]
        max_n_logits = 30  #How many logits to attribute from, max. We attribute to min(max_n_logits, n_logits_to_reach_des>
        desired_logit_prob = 0.95  #Attribution will attribute from the minimum number of logits needed to reach this proba>
        max_feature_nodes = 8192  #Only attribute from this number of feature nodes, max. Lower is faster, but you will los>
        batch_size = 64  #Batch size when attributing
        verbose = True 

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
            
            graph_dir = "graphs_adv_gemma"
            graph_name = category+ "_" + str(prompt_num) + ".pt"
            graph_dir = Path(graph_dir)
            graph_dir.mkdir(exist_ok=True)
            graph_path = graph_dir / graph_name

            graph.to_pt(graph_path)

            os.makedirs("in_graph_files_adv_gemma", exist_ok=True)
            nm = category + "_" + str(prompt_num)
            slug =  nm + "_slug" #this is the name that you assign to the graph
            graph_file_nm = nm + "_" + "graph_file"#TODO: change this one too.

            graph_file_dir = "./in_graph_files_adv_gemma/" + graph_file_nm  #where to write the graph files. no need to mak>
            node_threshold = 0.7 
            edge_threshold = 0.95  

            create_graph_files(
                graph_or_path=graph_path, 
                slug=slug,
                output_path=graph_file_dir,
                node_threshold=node_threshold,
                edge_threshold=edge_threshold,
            )
        except Exception as e:
            print(f"Error on prompt {prompt_num}: {e}")
        finally:
            torch.cuda.empty_cache()  
cleanup(model)
print("Finished gemma. Uncomment the llama section and run again.")
