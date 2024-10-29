import argparse
import torch
from transformers import AutoTokenizer, LlamaForCausalLM
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    return mask
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="meta-llama/Llama-2-7b-chat-hf",help='model')
parser.add_argument('--T', type=int, default=2000, help='repeat times')
parser.add_argument('--B', type=int, default=1, help='batch size')
parser.add_argument('--M', type=int, default=512, help='max length')
parser.add_argument('--D', type=int, default=1, help='dec length')
parser.add_argument('--G', type=int, default=64, help='generation length')
args = parser.parse_args()
print(args)
MAX_LEN = args.M
DEC_LEN = args.D
GEN_LEN = args.G
MODEL_NAME = args.model
DTYPE = torch.float16
DEVICE = "cuda:0"
T = args.T
WARM_UP = 10

llm = LlamaForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=DTYPE).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
text = "[INST] Hello, tell me what you know about China? [/INST]"
input_ids = tokenizer.encode(text=text, return_tensors="pt").to(device=DEVICE)
cache = None
with torch.inference_mode():
    output = llm(input_ids=input_ids, past_key_values=cache, use_cache=True)
    cache = output.past_key_values
    logits = output.logits
    print(logits)
    for i in range(GEN_LEN):
        input_ids = torch.argmax(logits[:,-1,:], keepdim=True)
        output = llm(input_ids=input_ids, past_key_values=cache, use_cache=True)
        cache = output.past_key_values
        logits = output.logits
        print(logits)
