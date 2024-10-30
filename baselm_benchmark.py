from baselm import LLMEngine
import argparse
import time
import torch
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
parser.add_argument('--model', type=str, default="Felladrin/Llama-68M-Chat-v1",help='model')
parser.add_argument('--T', type=int, default=2000, help='repeat times')
parser.add_argument('--B', type=int, default=1, help='batch size')
parser.add_argument('--P', type=int, default=128, help='prefix length')
parser.add_argument('--M', type=int, default=512, help='max length')
parser.add_argument('--D', type=int, default=1, help='dec length')
args = parser.parse_args()
print(args)
PREFIX_LEN = args.P
MAX_LEN = args.M
DEC_LEN = args.D
MODEL_NAME = args.model
DTYPE = torch.float16
DEVICE = "cuda:0"
T = args.T
WARM_UP = 10

llm = LLMEngine(max_length=MAX_LEN, model_name=args.model)
input_ids = torch.randint(low=3, high=30000, size=(1, PREFIX_LEN), device=DEVICE)
attention_mask = _make_causal_mask((MAX_LEN, MAX_LEN), dtype=DTYPE, device=DEVICE)
position_ids = torch.arange(PREFIX_LEN, device=DEVICE).unsqueeze(0)
prefix_storage_ids = torch.arange(PREFIX_LEN, device=DEVICE)
llm.initialize_cuda_graph([DEC_LEN, PREFIX_LEN])
llm.inference(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask[..., :PREFIX_LEN,:], storage_ids=prefix_storage_ids)

input_ids = torch.randint(low=3, high=30000, size=(1, DEC_LEN), device=DEVICE)
storage_ids = torch.arange(DEC_LEN, device=DEVICE) + PREFIX_LEN
position_ids = storage_ids.clone().unsqueeze(0)
attention_mask = attention_mask[..., PREFIX_LEN: PREFIX_LEN + DEC_LEN,:].clone()
for _ in range(WARM_UP):
    llm.inference(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask, storage_ids=storage_ids)

torch.cuda.synchronize()
t1 = time.time()
for _ in range(T):
    llm.inference(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask, storage_ids=storage_ids)
torch.cuda.synchronize()
t2 = time.time()

print("Max Length :{}, Decode Length :{}, Prefix Length :{}, inference time:{}s".format(MAX_LEN, DEC_LEN, PREFIX_LEN, (t2 - t1)/ T))