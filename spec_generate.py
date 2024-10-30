from speculation_engine import SpeculationEngine
import os
os.environ['TORCH_CUDA_ARCH_LIST'] =  "8.9"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="meta-llama/Llama-2-7b-chat-hf",help='model')
parser.add_argument('--draft_model', type=str, default="Felladrin/Llama-68M-Chat-v1",help='draft model')
parser.add_argument('--G', type=int, default=64, help='generation length')
parser.add_argument('--growmap', type=str, default="trees/L40-CNN-68m-7b-greedy.pt", help='growmap path')
args = parser.parse_args()
print(args)

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
    max_length=512,
    gen_length=128
)


text1 = """[INST] Hello, tell me what you know about China? [/INST] \n"""

text2 = """[INST] Hello, tell me the relationship between China and Japan? [/INST] \n"""


engine.initialize()
engine.prefill(text1)
engine.speculative_decoding()