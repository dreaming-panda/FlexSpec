from llm import LLMEngine
import argparse
import torch
from transformers import AutoTokenizer
import os
os.environ['TORCH_CUDA_ARCH_LIST'] =  "8.9"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(False, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), True)
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

llm = LLMEngine(max_length=MAX_LEN, model_name=args.model)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
text = """[INST] Hello, tell me what you know about China? [/INST] \n"""
input_ids = tokenizer.encode(text=text, return_tensors="pt").to(device=DEVICE)
# input_ids = torch.tensor([[    1,   518, 25580, 29962, 15043, 29892,  2649,   592,   825,   366,
#           1073,  1048,  7551, 29973,   518, 29914, 25580, 29962, 29871,    13, 10994]],
#        device='cuda:0').long()
PREFIX_LEN = input_ids.shape[1]
attention_mask = _make_causal_mask((MAX_LEN, MAX_LEN), dtype=DTYPE, device=DEVICE)
#attention_mask = attention_mask[None, None, :, :]
position_ids = torch.arange(MAX_LEN, device=DEVICE).unsqueeze(0)
prefix_storage_ids = torch.arange(MAX_LEN, device=DEVICE)
llm.initialize_cuda_graph([DEC_LEN, PREFIX_LEN])

logits = llm.inference(input_ids=input_ids, position_ids=position_ids[:,:PREFIX_LEN], attention_mask=attention_mask[:PREFIX_LEN,:], storage_ids=prefix_storage_ids[:PREFIX_LEN])
print(logits)
token = input_ids[0].tolist()
for i in range(GEN_LEN):
    #print(logits[:,-1,:])
    input_ids = torch.argmax(logits[:,-1,:], keepdim=True)
    #print(input_ids)
    print((logits[:,-1,:].topk(3)))
    logits = llm.inference(input_ids=input_ids, position_ids=position_ids[:,PREFIX_LEN+i:PREFIX_LEN+i+1], attention_mask=attention_mask[PREFIX_LEN+i:PREFIX_LEN+i+1,:], storage_ids=prefix_storage_ids[PREFIX_LEN+i:PREFIX_LEN+i+1])
    #print(logits)
