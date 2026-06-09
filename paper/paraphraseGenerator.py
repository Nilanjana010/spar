from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
import torch, gc, os
from parascore import ParaScorer
from tqdm import tqdm
import pandas as pd


device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "Vamsi/T5_Paraphrase_Paws"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
model.eval()

parascore_model = ParaScorer(lang="en", device=device)

dataset = load_dataset("izi-ano/CounselBench-Eval", split="test")
df = dataset.to_pandas()

#Create the merged input
df['combined_input'] = df['questionTitle'].fillna('') + ". " + df['questionText'].fillna('')

text_pool = df['combined_input'].unique()
print(f"Total unique merged texts to process: {len(text_pool)}")


def get_binned_paraphrases(original_text, candidates, scores):
    scored = list(zip(candidates, scores))
    scored.sort(key=lambda x: x[1], reverse=True) 
    
    bins = {}
    if not scored: return bins

    #Most similar (Low), Middle (Med), Most diverse (High)
    bins["Low"] = scored[0]
    bins["Medium"] = scored[len(scored)//2]
    bins["High"] = scored[-1]
    return bins

def process_batch(combined_texts, num_return=10):
    inputs = tokenizer(combined_texts, return_tensors="pt", padding=True, truncation=True).to(device)    
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        num_beams=15, 
        num_return_sequences=num_return,
        early_stopping=True
)
    
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    results_rows = []

    for i, original_combined in enumerate(combined_texts):
        #Ensure we don't just get the exact original back
        candidates = list(set([c for c in decoded[i * num_return : (i + 1) * num_return] if c.lower() != original_combined.lower()]))
        
        if not candidates: candidates = [original_combined]

        try:
            #Get the scores from ParaScore
            scores_tensor = parascore_model.score([original_combined] * len(candidates), candidates)
            
            if torch.is_tensor(scores_tensor):
                scores = scores_tensor.detach().cpu().flatten().tolist()

            elif isinstance(scores_tensor, list):
                scores = []
                for s in scores_tensor:
                    if torch.is_tensor(s):
                        scores.extend(s.detach().cpu().flatten().tolist())
                    elif isinstance(s, tuple):
                        val = s[0]
                        if torch.is_tensor(val):
                            scores.extend(val.detach().cpu().flatten().tolist())
                        else:
                            scores.append(float(val))
                    else:
                        scores.append(float(s))

            elif isinstance(scores_tensor, tuple):
                val = scores_tensor[0]
                if torch.is_tensor(val):
                    scores = val.detach().cpu().flatten().tolist()
                else:
                    scores = [float(val)] * len(candidates)

            else:
                scores = [float(scores_tensor)] * len(candidates)
                            
                #Verify the lengths match to prevent alignment issues
                if len(scores) != len(candidates):
                    print(f"Warning: Score length ({len(scores)}) mismatch with candidates ({len(candidates)})")
                    scores = scores[:len(candidates)] 
                
        except Exception as e:
            print(f"Scoring error: {e}")
            scores = [0.5] * len(candidates)
            binned = get_binned_paraphrases(original_combined, candidates, scores)

        for level, data in binned.items():
            para_text, score = data
            words = para_text.split()
            
            if len(words) > 1:
                body = " ".join(words[:-1])
                last_word = words[-1]
            else:
                body = ""
                last_word = para_text

            results_rows.append([original_combined, body, last_word, score, level])
                
    return results_rows

#Execution
batch_size = 4 
final_data = []

for i in tqdm(range(0, len(text_pool), batch_size)):
    batch = text_pool[i : i + batch_size].tolist()
    rows = process_batch(batch)
    final_data.extend(rows)
if i % 100 == 0:
        torch.cuda.empty_cache()

#Save Output
output_df = pd.DataFrame(final_data, columns=[
    'Original_Combined_Text', 
    'Paraphrase_Minus_Last_Word', 
    'Final_Word', 
    'ParaScore',
    'Paraphrase_Intensity'
])

output_df.to_csv("counselbench_eval_merged_3levels.csv", index=False)
print(f"Done! Created {len(output_df)} rows.")
