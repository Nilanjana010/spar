import os, pickle

from bleurt import score
checkpoint = "BLEURT-20"

gemma_dict = {}
for gem_file in os.listdir("validate_gemma"):
    print(gem_file)
    with open(os.path.join("validate_gemma", gem_file), "rb") as f:
        g_file = pickle.load(f)
        print("prompt", g_file[0], "\n\nPre", g_file[1][0][0], "\n\nPost", g_file[2][0][0])
        reference = g_file[1][0][0]
        candidate = g_file[2][0][0]
        p_len = len(g_file[0])
        if reference[:p_len] == candidate[:p_len] and reference[:p_len] == g_file[0]:
               reference = reference[p_len:]
               candidate = candidate[p_len:]
        scorer = score.BleurtScorer(checkpoint)
        scores = scorer.score(references=[reference], candidates=[candidate])
        assert isinstance(scores, list) and len(scores) == 1
        print(f'g_file = {g_file}, scores = {scores}')
        gemma_dict[gem_file] = scores

llama_dict = {}
for lla_file in os.listdir("validate_llama"):
    print(lla_file)
    with open(os.path.join("validate_llama", lla_file), "rb") as f:
        l_file = pickle.load(f)
        print("prompt", l_file[0], "\n\nPre", l_file[1][0][0], "\n\nPost", l_file[2][0][0])
        reference = l_file[1][0][0]
        candidate = l_file[2][0][0]
        p_len = len(l_file[0])
        if reference[:p_len] == candidate[:p_len] and reference[:p_len] == l_file[0]:
               reference = reference[p_len:]
               candidate = candidate[p_len:]
        scorer = score.BleurtScorer(checkpoint)
        scores = scorer.score(references=[reference], candidates=[candidate])
        assert isinstance(scores, list) and len(scores) == 1
        print(f'l_file = {l_file}, scores = {scores}')
        llama_dict[lla_file] = scores

with open("gemma_bleurt_scores", "wb") as f:
            pickle.dump(gemma_dict, f)
with open("llama_bleurt_scores", "wb") as f:
            pickle.dump(llama_dict, f)


