import os
import json
import pickle
from collections import namedtuple, defaultdict
from datasets import load_dataset

import torch
from huggingface_hub import login

from circuit_tracer import ReplacementModel
from circuit_tracer.utils.demo_utils import display_generations_comparison


backend = 'transformerlens'
transcoder_name = 'mntss/clt-gemma-2-2b-426k'
model_name = 'google/gemma-2-2b'

login(token="******")

model = ReplacementModel.from_pretrained(
    model_name,
    transcoder_name,
    dtype=torch.bfloat16,
    backend=backend,
)

Feature = namedtuple('Feature', ['layer', 'pos', 'feature_idx'])

os.makedirs('validate_gemma_advbench', exist_ok=True)

exp_list = [0.8, 0.9, 0.95, 0.97, 1.0]
gemma_eval_fol = 'sparsified_gemma_advbench'
NUM_LAYERS = 26
NUM_FEATURES = 16384

ds = load_dataset("walledai/AdvBench")

unique_prompts = []

for ds_load in range(len(ds['train'])):
     unique_prompts.append(ds['train'][ds_load]['prompt'])

def ret_neurons(json_data):
    selected = defaultdict(set)
    original_nodes = json_data.get('nodes', [])

    print(f"Input Stats: {len(original_nodes)} nodes")

    for node in original_nodes:
        if node.get('feature_type') != 'cross layer transcoder':
            continue

        layer = int(node['layer'])
        if layer not in range(NUM_LAYERS):
            continue

        n_id = str(node.get('id') or node.get('node_id') or node.get('name'))
        id_split = n_id.split('_')
        if len(id_split) != 3:
            continue

        extracted_layer, extracted_feature, extracted_ctx = id_split
        if extracted_layer != str(layer):
            continue

        feature_idx = int(extracted_feature)
        if feature_idx < 0 or feature_idx >= NUM_FEATURES:
            continue

        ctx_idx = int(node.get('ctx_idx', extracted_ctx))
        if ctx_idx != int(extracted_ctx):
            continue

        selected[(layer, ctx_idx)].add(feature_idx)

    return selected


def generation(prompt, selected_nodes, exponent, prompt_no):
    intervention_tuples = []
    tokenized = model.tokenizer(prompt)
    sequence_length = len(tokenized.input_ids)
    print("sequence length = ", sequence_length)

    for (layer, ctx_idx), keep_features in selected_nodes.items():
        if ctx_idx < 0 or ctx_idx >= sequence_length:
            continue

        for feature_idx in range(NUM_FEATURES):
            if feature_idx not in keep_features:
                intervention_tuples.append((layer, ctx_idx, feature_idx, 0.0))

    print('num_interventions = ', len(intervention_tuples))

    pre_intervention_generation = [
        model.feature_intervention_generate(prompt, [], do_sample=False, verbose=False)[0]
    ]
    post_intervention_generation = [
        model.feature_intervention_generate(
            prompt,
            intervention_tuples,
            do_sample=False,
            verbose=False,
        )[0]
    ]

    print('pre_intervention_generation = ', pre_intervention_generation)
    print('post_intervention_generation = ', post_intervention_generation)

    displayed_res = display_generations_comparison(
        prompt,
        pre_intervention_generation,
        post_intervention_generation,
    )
    print('displayed results = ', displayed_res)

    obj = (prompt, [pre_intervention_generation], [post_intervention_generation])
    save_file = 'generations_advbench_obj_gemma_e' + str(exponent) + '_prompt_no_' + str(prompt_no) + '.pkl'
    with open(os.path.join('validate_gemma_advbench', save_file), 'wb') as f:
        pickle.dump(obj, f)

for gemma_file in os.listdir(gemma_eval_fol):
    if not gemma_file.endswith('.json'):
        continue
    print("gemma_file = ", gemma_file)
    value = float(gemma_file.split('_slug_e')[1].replace('.json', ''))
    print('value = ', value)
    if value in exp_list:
        with open(os.path.join(gemma_eval_fol, gemma_file), 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        extract_prompt_no = int(gemma_file.split('_')[1])
        print('extracted prompt number = ', extract_prompt_no)
        get_neurons = ret_neurons(json_data)
        print('num_selected_layer_ctx = ', len(get_neurons))
        generation(unique_prompts[extract_prompt_no], get_neurons, value, extract_prompt_no)