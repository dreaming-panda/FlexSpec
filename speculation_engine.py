import torch
from llm import LLMEngine, _make_causal_mask
from transformers import AutoTokenizer
from utils import cuda_graph_for_sampling_argmax
import time
import flashinfer
class SpeculationEngine:

    def __init__(self,
        draft_model_name: str,
        target_model_name: str,
        growmap_path: str,
        dtype=torch.float16,
        max_length=512,
        gen_length=128,
        temperature=0,
        topp = 0.9,
        device :str = 'cuda:0',
        vocab_size = 32000,
        ) -> None:
        
        self.max_length = max_length
        self.gen_length = gen_length
        self.draft_model_name = draft_model_name
        self.target_model_name = target_model_name
        self.dtype = dtype
        self.device = device
        self.temperature = temperature
        self.topp = topp
        self.growmap_path = growmap_path
        self.vocab_size = vocab_size

        
    def initialize(self):
        self.growmap = torch.load(self.growmap_path)
        self.storage_ids = torch.arange(self.max_length).to(self.device)
        self.tokens = torch.zeros(1, self.max_length, device=self.device).long()
        
        self.Successors :list[list[int]] = []
        self.attn_mask = torch.full((2 * self.max_length, 2 * self.max_length), False, dtype=torch.bool, device=self.device)
        self.position_ids = torch.zeros(1, self.max_length).long().to(self.device)
        self.attn_mask[:self.max_length, :self.max_length] = _make_causal_mask((1, self.max_length),dtype=torch.bool, device=self.device)
        
        idx_lists = self.growmap["roots"]
        self.draft_step = len(idx_lists)
        self.accept_position = torch.zeros(self.draft_step, dtype=torch.int32, device=self.device)
        self.growmap_roots = []
        for x in idx_lists:
             self.growmap_roots.append(torch.Tensor(x).to(self.device).long())
        self.Successors = self.growmap["Successors"]
        tree_mask :torch.Tensor = self.growmap["mask"].to(self.device)
        tree_mask = (tree_mask == 1)
        self.tree_size = self.growmap["size"]
        print("[Sequoia], Tree Size {} | Tree Depth {}".format(self.tree_size, self.draft_step))
        
        
        self.attn_mask[self.max_length - self.tree_size: self.max_length, self.max_length - self.tree_size: self.max_length] = tree_mask
        
        
        self.depth = self.growmap["depth"].to(self.device)
        
        self.branch_lists = self.growmap['branches']
        graph_capture_list = [sum(x) for x in self.branch_lists if sum(x) > 0]
        graph_capture_list.append(1)
        
        self.draft_logits = torch.zeros((self.max_length, self.vocab_size), dtype=torch.float32).to(self.device)
        self.target_logits = torch.zeros((self.max_length, self.vocab_size), dtype=torch.float32).to(self.device)
        self.draft_model = LLMEngine(
                    self.draft_model_name, batch_size=1, 
                    max_length=self.max_length, device=self.device,
                    dtype=self.dtype)
        
        self.target_model = LLMEngine(
                    self.target_model_name, batch_size=1, 
                    max_length=self.max_length, device=self.device,
                    dtype=self.dtype)
        
        self.draft_model.initialize_cuda_graph(graph_capture_list)
        print("[DRAFT MODEL]: Initialize CUDA GRAPH")
        self.target_model.initialize_cuda_graph([self.tree_size])
        print("[TARGET MODEL]: Initialize CUDA GRAPH")
        
        
        self.sampling_callables = {}
        self.sample_gather_indices = {}
        
        for i in range(self.draft_step - 1):
            idx_len = len(idx_lists[i])
            num_samples = max(self.branch_lists[i])
            self.sampling_callables[i] = cuda_graph_for_sampling_argmax(device=self.device, idx_len=idx_len, num_samples=num_samples)
        
        for i in range(self.draft_step - 1):
            ith_gather_list = []
            max_num_samples = max(self.branch_lists[i])
            for j, branch in enumerate(self.branch_lists[i]):
                branch_index = torch.arange(branch, device=self.device, dtype=torch.long)
                branch_index = branch_index + j * max_num_samples
                ith_gather_list.append(branch_index)
            ith_gather_list = torch.cat(ith_gather_list)
            self.sample_gather_indices[i] = ith_gather_list
        
        print("[SAMPLER]: Initialize CUDA GRAPH")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.target_model_name)
        self.num_nodes = 0
    def prefill(self, text:str):
        input_ids = self.tokenizer.encode(text=text, return_tensors="pt").to(device=self.device)
        
        self._prefill(input_ids=input_ids)
    
    def _prefill(self, input_ids:torch.LongTensor):
        
        prefix_len = input_ids.shape[1]
        self.prefix_len = prefix_len
        self.num_nodes += prefix_len
        self.num_nodes_this_iter = self.num_nodes + self.tree_size
        self.attn_mask_this_iter = self.attn_mask[self.max_length - self.num_nodes_this_iter: 2 * self.max_length - self.num_nodes_this_iter, self.max_length - self.num_nodes_this_iter: 2 * self.max_length - self.num_nodes_this_iter].contiguous()
        self.num_draft_tokens_this_iter = self.num_nodes
        
        self.tokens[:,:prefix_len].copy_(input_ids)
        
        self.position_ids[:,:prefix_len] = torch.arange(prefix_len).unsqueeze(0)
        self.position_ids[:,prefix_len:prefix_len+self.tree_size] = prefix_len + self.depth
        
        draft_logits = self.draft_model.prefill(
             input_ids=self.tokens[:,:prefix_len],
             storage_ids=self.storage_ids[:prefix_len],
             position_ids=self.position_ids[:,:prefix_len],
             attention_mask=self.attn_mask_this_iter[:prefix_len]
        )[0]
    
        self.draft_logits[:prefix_len].copy_(draft_logits)
        
        
        
        target_logits = self.target_model.prefill(
             input_ids=self.tokens[:,:prefix_len],
             storage_ids=self.storage_ids[:prefix_len],
             position_ids=self.position_ids[:,:prefix_len],
             attention_mask=self.attn_mask_this_iter[:prefix_len]
        )[0]

        self.target_logits[:prefix_len].copy_(target_logits)
        
        next_token = target_logits[-1:].argmax(dim=-1, keepdim=True)
        
        self.tokens[:,self.num_nodes:self.num_nodes+1] = next_token
        
    def speculative_decoding(self):
        t1 = time.time()
        large_model_step = 0
        decode = True
        while (self.num_nodes - self.prefix_len < self.gen_length) and decode:
            self.build_tree()
            decode = self.verify()
            large_model_step = large_model_step + 1
        t2 = time.time()
        tokens = self.tokens[0,:self.num_nodes + 1].tolist()
        text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        dec_len = len(tokens) - self.prefix_len
        print(text)
        print("[Speculative Decoding]: Avg Accept Tokens {:.2f}".format(dec_len/large_model_step))
        print("[Speculative Decoding]: TPOT {:.2f} ms".format(1000 * (t2-t1)/dec_len))
        return dec_len, (t2 - t1), large_model_step
    def build_tree(self):
        
        for step in range(self.draft_step):
            idx_list = self.growmap_roots[step]
            branch_list = self.branch_lists[step]
            total_branch = sum(branch_list)
            dec_len = len(idx_list)
            
            input_ids = self.tokens[:,self.num_draft_tokens_this_iter:self.num_draft_tokens_this_iter+dec_len]
            
            position_ids = self.position_ids[:,self.num_draft_tokens_this_iter:self.num_draft_tokens_this_iter+dec_len]
            storage_ids = self.storage_ids[self.num_draft_tokens_this_iter:self.num_draft_tokens_this_iter+dec_len]
            attention_mask = self.attn_mask_this_iter[self.num_draft_tokens_this_iter:self.num_draft_tokens_this_iter+dec_len]
            draft_logits = self.draft_model.inference(
            input_ids=input_ids,
            storage_ids=storage_ids,
            position_ids=position_ids,
            attention_mask=attention_mask
            )[0]
    
            self.draft_logits[self.num_draft_tokens_this_iter:self.num_draft_tokens_this_iter+dec_len].copy_(draft_logits)
            self.num_draft_tokens_this_iter += dec_len
            if step < self.draft_step - 1:
                
                new_tokens_set = self.sampling_callables[step](draft_logits)
                self.tokens[0,self.num_draft_tokens_this_iter: self.num_draft_tokens_this_iter + total_branch] = new_tokens_set[self.sample_gather_indices[step]]
        
    def verify(self):
        
        input_ids = self.tokens[:,self.num_nodes:self.num_draft_tokens_this_iter]
        position_ids = self.position_ids[:,self.num_nodes:self.num_draft_tokens_this_iter]
        storage_ids = self.storage_ids[self.num_nodes:self.num_draft_tokens_this_iter]
        attention_mask = self.attn_mask_this_iter[self.num_nodes:self.num_draft_tokens_this_iter]
        
        target_logits = self.target_model.inference(
             input_ids=input_ids,
             storage_ids=storage_ids,
             position_ids=position_ids,
             attention_mask=attention_mask
        )[0]
        
        
        num_accept_tokens = 0
        if self.temperature < 0.05:
            # greedy decoding
            sampled_tokens = target_logits.argmax(dim=-1)
        
        else:
            #stochastic decoding
            proba = torch.softmax(target_logits/self.temperature, dim=-1)
            proba = flashinfer.sampling.top_p_renorm_probs(proba, self.topp)
            sampled_tokens = torch.multinomial(proba, num_samples=1).squeeze(-1)
            
        speculated_tokens = self.tokens[0, self.num_nodes:self.num_draft_tokens_this_iter]
        self.accept_position.zero_()
        terminal = False
        while not terminal:
                parent_id = self.accept_position[num_accept_tokens]
                target_token = sampled_tokens[parent_id]
                children = self.Successors[parent_id]
                if len(children) == 0:
                    terminal = True
                else:
                    terminal = True
                    for pos in children:
                        token = speculated_tokens[pos]
                        if token == target_token:
                            num_accept_tokens += 1
                            self.accept_position[num_accept_tokens] = pos
                            terminal = False
                            break
        accept_length = num_accept_tokens + 1
        accept_tokens = speculated_tokens[self.accept_position[:accept_length]]
        self.tokens[0, self.num_nodes:self.num_nodes + accept_length] = accept_tokens
        self.tokens[0, self.num_nodes + accept_length] = target_token
        
        if ((self.tokens[0, self.num_nodes:self.num_nodes + accept_length+1] == 0)._is_any_true() or
        (self.tokens[0, self.num_nodes:self.num_nodes + accept_length+1] == 2)._is_any_true()
        ):
            return False
          
        self.accept_position += self.num_nodes
        self.draft_model.llm.kv_cache.gather_kv_incremental(self.accept_position[:accept_length], self.num_nodes)
        self.target_model.llm.kv_cache.gather_kv_incremental(self.accept_position[:accept_length], self.num_nodes)
        
        num_last_iter_nodes = self.num_nodes
        self.num_nodes = self.num_nodes + accept_length
        self.num_draft_tokens_this_iter = self.num_nodes
        
        self.num_nodes_this_iter = self.num_nodes + self.tree_size
        self.attn_mask_this_iter = self.attn_mask[self.max_length - self.num_nodes_this_iter: 2 * self.max_length - self.num_nodes_this_iter, self.max_length - self.num_nodes_this_iter: 2 * self.max_length - self.num_nodes_this_iter].contiguous()
        
        self.position_ids[:,num_last_iter_nodes:self.num_nodes] = self.position_ids[:,self.accept_position[:accept_length]]
        self.position_ids[:,self.num_nodes:self.num_nodes+self.tree_size] = self.num_nodes + self.depth
        
        return True
    def reset(self):
        self.prefix_len = 0
        self.num_nodes = 0
        self.tokens.zero_()
        self.position_ids.zero_()
        self.draft_logits.zero_()
        self.target_logits.zero_()
        self.accept_position.zero_()
        self.draft_model.llm.kv_cache.clear()
        self.target_model.llm.kv_cache.clear()
        