import json
import os.path as osp
import ssl
import urllib.request
import os
from baselm import LLMEngine
from llm import _make_causal_mask
from transformers import AutoTokenizer
import torch
import time
os.environ['TORCH_CUDA_ARCH_LIST'] =  "8.9"
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
MAX_LEN = 512
DTYPE = torch.float16
engine = LLMEngine(model_name=MODEL_NAME, batch_size=1, max_length=MAX_LEN, device=DEVICE,dtype=torch.float16)
engine.initialize_cuda_graph([1])
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


attention_mask = _make_causal_mask((MAX_LEN, MAX_LEN), dtype=DTYPE, device=DEVICE)
position_ids = torch.arange(MAX_LEN, device=DEVICE).unsqueeze(0)
prefix_storage_ids = torch.arange(MAX_LEN, device=DEVICE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
total_time = 0
total_decode_tokens = 0
for idx, prompt in enumerate(prompts[:10]):
    input_ids = tokenizer.encode(text=prompt, return_tensors="pt").to(DEVICE)
    prefix_len = input_ids.shape[1]
    if prefix_len > 128:
        continue
    else:
        logits = engine.inference(input_ids=input_ids, position_ids=position_ids[:,:prefix_len], attention_mask=attention_mask[:prefix_len,:], storage_ids=prefix_storage_ids[:prefix_len])
        token = torch.zeros(MAX_LEN).to(DEVICE).long()
        dec_tokens = 0
        token[:prefix_len] = input_ids[0]
        torch.cuda.synchronize()
        t1 = time.time()
        for i in range(GEN_LEN):
            input_ids = torch.argmax(logits[:,-1,:], keepdim=True)
            logits = engine.inference(input_ids=input_ids, position_ids=position_ids[:,prefix_len+i:prefix_len+i+1], attention_mask=attention_mask[prefix_len+i:prefix_len+i+1,:], storage_ids=prefix_storage_ids[prefix_len+i:prefix_len+i+1])
            token[prefix_len+i] = input_ids
            dec_tokens+=1
            if input_ids[0].item() in [0,2]:
                break
        torch.cuda.synchronize()
        t2 = time.time()
        text = tokenizer.decode(token.tolist(), skip_special_tokens=True)
        print(text)

        total_decode_tokens += dec_tokens
        total_time += (t2 - t1)
        engine.llm.kv_cache.clear()
        
print("[Summary | Speculative Decoding]: TPOT {:.2f} ms".format(1000 * total_time/total_decode_tokens))
