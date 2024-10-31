import json
import os.path as osp
import ssl
import urllib.request
import os
from speculation_engine import SpeculationEngine
import torch
os.environ['TORCH_CUDA_ARCH_LIST'] =  "8.0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="meta-llama/Llama-2-7b-chat-hf",help='model')
parser.add_argument('--draft_model', type=str, default="Felladrin/Llama-68M-Chat-v1",help='draft model')
parser.add_argument('--G', type=int, default=128, help='generation length')
parser.add_argument('--growmap', type=str, default="trees/L40-CNN-68m-7b-greedy.pt", help='growmap path')
args = parser.parse_args()
print(args)
torch.cuda.set_device(0)
MODEL_NAME = args.model
DEVICE = "cuda:0"
GEN_LEN = args.G
path = args.growmap
draft_model_name = args.draft_model
target_model_name = args.model

engine = SpeculationEngine(
    draft_model_name=draft_model_name,
    target_model_name=target_model_name,
    growmap_path=path,
    device=DEVICE,
    max_length=2048,
    gen_length=GEN_LEN
)
engine.initialize()

def download_url(url: str, folder="folder"):
    """
    Downloads the content of an url to a folder. Modified from \
    https://github.com/pyg-team/pytorch_geometric/tree/master/torch_geometric

    Args:
        url (string): The url of target file.
        folder (string): The target folder.

    Returns:
        string: File path of downloaded files.
    """

    file = url.rpartition("/")[2]
    file = file if file[0] == "?" else file.split("?")[0]
    path = osp.join(folder, file)
    if osp.exists(path):
        print(f"File {file} exists, use existing file.")
        return path

    print(f"Downloading {url}")
    os.makedirs(folder, exist_ok=True)
    ctx = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=ctx)
    with open(path, "wb") as f:
        f.write(data.read())

    return path


def load_jsonl(
    file_path,
):
    list_data_dict = []
    with open(file_path, "r") as f:
        for line in f:
            list_data_dict.append(json.loads(line))
    return list_data_dict

test_filepath = os.path.join("data/", "question.jsonl")
print(f"Loading data from {test_filepath} ...")

if not os.path.exists(test_filepath):
    download_url(
            "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl",
            "data/",
        )
    os.rename(os.path.join("data/", "question.jsonl"), test_filepath)

list_data = load_jsonl(test_filepath)
prompts = []
for sample in list_data:
    prompts.append(sample["turns"][0])


large_model_steps = 0
total_time = 0
total_decode_tokens = 0
for idx, prompt in enumerate(prompts):
    input_ids = engine.tokenizer.encode(text=prompt, return_tensors="pt")
    if input_ids.shape[1] > 128:
        continue
    else:
        print(idx)
        engine.prefill(prompt)
        
        num_tokens, decode_time, step = engine.speculative_decoding()
        total_time += decode_time
        total_decode_tokens += num_tokens
        large_model_steps += step
        engine.reset()

print("[Summary | Speculative Decoding]: Avg Accept Tokens {:.2f}".format(total_decode_tokens/large_model_steps))
print("[Summary | Speculative Decoding]: TPOT {:.2f} ms".format(1000 * total_time/total_decode_tokens))
