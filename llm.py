from transformers import LlamaForCausalLM, LlamaConfig
import torch
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
import torch.nn.functional as F
from typing import List, Optional
import gc
import flashinfer
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

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
    
def capture_cuda_graph_for_pos_emb(
    bsz: int,
    q_len: int,
    num_head: int,
    num_kv_head: int,
    head_dim:int,
    max_len: int,
    dtype= torch.float16,
    device= "cuda:0",
    n_warmups=3, mempool=None
):
    static_q = torch.zeros((bsz, num_head, q_len, head_dim), dtype=dtype, device=device)
    static_k = torch.zeros((bsz, num_kv_head, q_len, head_dim), dtype=dtype, device=device)
    static_sin = torch.zeros((max_len, head_dim), dtype=dtype, device=device)
    static_cos = torch.zeros((max_len, head_dim), dtype=dtype, device=device)
    static_pos = torch.zeros((bsz, q_len), dtype=torch.int32, device=device)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(n_warmups):
            new_q, new_k = apply_rotary_pos_emb(
                    static_q,
                    static_k,
                    static_cos,
                    static_sin,
                    static_pos
                    )
        s.synchronize()
    torch.cuda.current_stream().wait_stream(s)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=mempool):
         new_q, new_k = apply_rotary_pos_emb(
                    static_q,
                    static_k,
                    static_cos,
                    static_sin,
                    static_pos
                    )
    def run(q, k, cos, sin, pos):
        static_q.copy_(q)
        static_k.copy_(k)
        static_cos.copy_(cos)
        static_sin.copy_(sin)
        static_pos.copy_(pos)
        graph.replay()
        return new_q.clone(), new_k.clone()
    
    return run

class KV_Cache:

    def __init__(self, 
        config :LlamaConfig,
        batch_size :int = 1,
        max_length :int = 256, 
        device :str = 'cuda:0',
        dtype = torch.float16) -> None:
        self.config = config
        self.max_length = max_length
        self.device = device
        self.dtype = dtype
        self.k_cache = torch.zeros(
            config.num_hidden_layers,
            max_length,
            config.num_key_value_heads,
            config.hidden_size // config.num_attention_heads,
            device=self.device,
            dtype=self.dtype
        )

        self.v_cache = torch.zeros(
            config.num_hidden_layers,
            max_length,
            config.num_key_value_heads,
            config.hidden_size // config.num_attention_heads,
            device=self.device,
            dtype=self.dtype
        )
        self.num_layers = config.num_hidden_layers
        self.kv_offset = 0

    def initialize_kv(self,
            k_cache :torch.Tensor,
            v_cache :torch.Tensor,
            kv_len :int):
        
        self.k_cache[...,:kv_len,:,:] = k_cache[...,:kv_len,:,:]
        self.v_cache[...,:kv_len,:,:] = v_cache[...,:kv_len,:,:]

        self.kv_offset = kv_len
        
        
    
    def gather_kv(self, indices: list[int]):

        self.k_cache[..., :len(indices), :,:] = self.k_cache[..., indices, :,:]
        self.v_cache[..., :len(indices), :,:] = self.v_cache[..., indices, :,:]

        self.k_cache[..., len(indices):, :,:] = 0.0
        self.v_cache[..., len(indices):, :,:] = 0.0

        self.kv_offset = len(indices)
    
    def gather_kv_incremental(self, indices: torch.LongTensor, offset:int):

        self.k_cache[:,offset:offset + len(indices), :,:] = self.k_cache[:,indices, :,:]
        self.v_cache[:,offset:offset + len(indices), :,:] = self.v_cache[:,indices, :,:]

        self.k_cache[:,offset + len(indices):, :,:] = 0.0
        self.v_cache[:,offset + len(indices):, :,:] = 0.0

        self.kv_offset = offset + len(indices)


    
    def update_kv_cache(self, 
            new_k_cache :torch.Tensor,
            new_v_cache :torch.Tensor,
            layer_idx :int,
            storage_ids: torch.LongTensor
            ):

        new_kv_len = storage_ids.shape[0]
        if layer_idx == 0:
            self.kv_offset += new_kv_len
        self.k_cache[layer_idx][self.kv_offset - new_kv_len:self.kv_offset] = new_k_cache
        self.v_cache[layer_idx][self.kv_offset - new_kv_len:self.kv_offset] = new_v_cache
       # self.k_cache[layer_idx].index_copy_(dim=0, index=storage_ids, source=new_k_cache)
       # self.v_cache[layer_idx].index_copy_(dim=0, index=storage_ids, source=new_v_cache)
        return self.k_cache[layer_idx][:self.kv_offset], self.v_cache[layer_idx][:self.kv_offset]

    def clear(self):
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.kv_offset = 0
    
    def get_usable_length(self, layer_idx:int, input_length :int):
            if layer_idx == self.num_layers - 1:
                return self.kv_offset
            else:
                return self.kv_offset + input_length
    
    def set_kv_len(self, kv_len :int):
            self.kv_offset = kv_len

def layer_norm(
    hidden_states: torch.Tensor,
    layernorm_variance_epsilon: float,
    layernorm_weight: torch.Tensor,
):  
    b, s, h = hidden_states.shape
    
    hidden_states = hidden_states.reshape(b * s, h)
    hidden_states = flashinfer.rmsnorm(hidden_states, layernorm_weight, layernorm_variance_epsilon)
    hidden_states = hidden_states.reshape(b, s, h)
    return hidden_states

class LLMLayer:
    def __init__(self, layer_idx) -> None:
        
        self.wq :torch.Tensor = None
        self.wk :torch.Tensor = None
        self.wv :torch.Tensor = None
        self.wo :torch.Tensor = None

        self.gate_proj :torch.Tensor = None 
        self.up_proj :torch.Tensor = None
        self.down_proj :torch.Tensor = None

        self.input_layernorm_weight :torch.Tensor = None
        self.input_layernorm_variance_epsilon :float = 0.0

        self.post_attention_layernorm_weight :torch.Tensor = None
        self.post_attention_layernorm_variance_epsilon :float = 0.0

        self.cos_cache :torch.Tensor = None
        self.sin_cache :torch.Tensor = None

        self.layer_idx = layer_idx
    
    def init_parameters(self, hf_layer: LlamaDecoderLayer):

        self.wq :torch.Tensor= hf_layer.self_attn.q_proj.weight.detach()
        self.wk :torch.Tensor= hf_layer.self_attn.k_proj.weight.detach()
        self.wv :torch.Tensor= hf_layer.self_attn.v_proj.weight.detach()
        self.wo :torch.Tensor= hf_layer.self_attn.o_proj.weight.detach()

        self.gate_proj = hf_layer.mlp.gate_proj.weight.detach()
        self.up_proj = hf_layer.mlp.up_proj.weight.detach()
        self.down_proj = hf_layer.mlp.down_proj.weight.detach()

        self.input_layernorm_weight = hf_layer.input_layernorm.weight.detach()
        self.input_layernorm_variance_epsilon = hf_layer.input_layernorm.variance_epsilon

        self.post_attention_layernorm_weight = hf_layer.post_attention_layernorm.weight.detach()
        self.post_attention_layernorm_variance_epsilon = hf_layer.post_attention_layernorm.variance_epsilon

        self.cos_cache :torch.Tensor= hf_layer.self_attn.rotary_emb.cos_cached
        self.sin_cache :torch.Tensor= hf_layer.self_attn.rotary_emb.sin_cached
    
    def init_gpu(self, device:str = 'cuda:0'):

        self.input_layernorm_weight = self.input_layernorm_weight.to(device, non_blocking=True)
        self.post_attention_layernorm_weight = self.post_attention_layernorm_weight.to(device, non_blocking=True)
        self.wq = self.wq.to(device, non_blocking=True)
        self.wk = self.wk.to(device, non_blocking=True)
        self.wv = self.wv.to(device, non_blocking=True)
        self.wo = self.wo.to(device, non_blocking=True)
        self.gate_proj = self.gate_proj.to(device, non_blocking=True)
        self.up_proj = self.up_proj.to(device, non_blocking=True)
        self.down_proj =  self.down_proj.to(device, non_blocking=True)



class LLM:
    def __init__(self, 
        model_name: str,
        batch_size :int = 1,
        max_length :int = 256, 
        device :str = 'cuda:0',
        dtype = torch.float16) -> None:
        
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self.config = LlamaConfig.from_pretrained(model_name)
        self.model_name = model_name
        self.max_length = max_length
        self.kv_cache = KV_Cache(self.config, max_length=max_length, device=device, dtype=dtype, batch_size=self.batch_size)
        self.init_parameters()
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = self.config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = self.config.max_position_embeddings
        self.rope_theta = self.config.rope_theta
        self.rope_callables =  {}
        self.mempool = None

    def init_parameters(self):

        hf_model = LlamaForCausalLM.from_pretrained(self.model_name, torch_dtype=self.dtype)
        self.embed_tokens = hf_model.model.embed_tokens.weight.detach().to(self.device)
        self.lm_head = hf_model.lm_head.weight.detach().to(self.device)

        self.norm_weight = hf_model.model.norm.weight.detach().to(self.device)
        self.norm_variance_epsilon = hf_model.model.norm.variance_epsilon

        self.cos_cache = hf_model.model.layers[0].self_attn.rotary_emb.cos_cached.to(self.device)[:self.max_length].to(self.dtype)
        self.sin_cache = hf_model.model.layers[0].self_attn.rotary_emb.sin_cached.to(self.device)[:self.max_length].to(self.dtype)
        self.layers :list[LLMLayer] = []
        
        for idx, hf_layer in enumerate(hf_model.model.layers):
            layer = LLMLayer(idx)
            layer.init_parameters(hf_layer=hf_layer)
            layer.init_gpu(self.device)
            self.layers.append(layer)
            hf_model.model.layers[idx] = None
            gc.collect()
            
        self.num_layers = len(self.layers)

    @torch.inference_mode()
    def initialize_cuda_graph(self, 
            decoding_seqlens :List[int],
            n_warmups=3):
        gc.collect()
        self.mempool = torch.cuda.graphs.graph_pool_handle()
        for decoding_seqlen in decoding_seqlens:
            if decoding_seqlen not in self.rope_callables and decoding_seqlen !=0:
                self.rope_callables[decoding_seqlen] = capture_cuda_graph_for_pos_emb(
                    bsz = self.batch_size,
                    q_len = decoding_seqlen,
                    num_head=self.num_heads,
                    num_kv_head=self.num_key_value_heads,
                    head_dim=self.head_dim,
                    max_len=self.max_length,
                    dtype=self.dtype,
                    device=self.device,
                    n_warmups=n_warmups,
                    mempool=self.mempool
                )
    def pre_attention_compute(
        self,
        hidden_states: torch.Tensor,
        input_layernorm_variance_epsilon: float,
        input_layernorm_weight: torch.Tensor,
        wq:torch.Tensor,
        wk:torch.Tensor,
        wv:torch.Tensor,
        num_heads:int,
        num_key_value_heads:int,
        head_dim:int
    ):  
        hidden_states = layer_norm(hidden_states, input_layernorm_variance_epsilon, input_layernorm_weight)
        bsz, q_len, _ = hidden_states.size()
        query_states = F.linear(hidden_states, wq)
        key_states = F.linear(hidden_states, wk)
        value_states = F.linear(hidden_states, wv)
        query_states = query_states.view(q_len, num_heads, head_dim)
        key_states = key_states.view(q_len, num_key_value_heads, head_dim)
        value_states = value_states.view(q_len, num_key_value_heads, head_dim)
        return query_states, key_states, value_states
    def post_attention_compute(
        self,
        attn_output: torch.Tensor,
        residual: torch.Tensor,
        post_attention_layernorm_variance_epsilon: float,
        post_attention_layernorm_weight: torch.Tensor,
        wo: torch.Tensor,
        gate_proj: torch.Tensor,
        up_proj: torch.Tensor,
        down_proj: torch.Tensor,
    ):  
    
    
        hidden_states = F.linear(attn_output, wo)
            
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = layer_norm(hidden_states, post_attention_layernorm_variance_epsilon, post_attention_layernorm_weight)
        up = F.linear(hidden_states, up_proj)
        gate = F.linear(hidden_states, gate_proj)
        gate = F.silu(gate)
        hidden_states = gate * up
        hidden_states = F.linear(hidden_states, down_proj)
        hidden_states = residual + hidden_states
        return hidden_states
    @torch.inference_mode()
    def layer_compute(self, 
            buffer: LLMLayer,
            layer_idx :int, 
            hidden_states: torch.FloatTensor, 
            position_ids: torch.LongTensor, 
            attention_mask: torch.FloatTensor,
            storage_ids: torch.LongTensor):

        residual = hidden_states
        bsz, q_len, _ = hidden_states.size()
        query_states, key_states, value_states = self.pre_attention_compute(
            hidden_states,
            buffer.input_layernorm_variance_epsilon,
            buffer.input_layernorm_weight,
            buffer.wq,
            buffer.wk,
            buffer.wv,
            self.num_heads,
            self.num_key_value_heads,
            self.head_dim
        )
        flashinfer.apply_rope_pos_ids_inplace(query_states, key_states, position_ids, interleave=False, rope_theta=1e4)
        key_states, value_states = self.kv_cache.update_kv_cache(key_states, value_states, layer_idx, storage_ids)        
        hidden_states = flashinfer.single_prefill_with_kv_cache(
                q = query_states,
                k = key_states,
                v = value_states,
                kv_layout="NHD",
                custom_mask=attention_mask[:,:key_states.shape[0]],
                allow_fp16_qk_reduction=True
            )
        
        hidden_states = hidden_states.reshape(bsz, q_len, self.hidden_size)
        
        hidden_states = self.post_attention_compute(
                        hidden_states, residual,
                        buffer.post_attention_layernorm_variance_epsilon,
                        buffer.post_attention_layernorm_weight,
                        buffer.wo,
                        buffer.gate_proj,
                        buffer.up_proj,
                        buffer.down_proj,
                        )
        
        return hidden_states


    @torch.inference_mode()
    def inference(self,
            input_ids: torch.LongTensor,
            position_ids: torch.LongTensor,
            attention_mask: torch.FloatTensor,
            storage_ids: torch.LongTensor):
        
        hidden_states = F.embedding(input_ids, self.embed_tokens)  
        for idx in range(self.num_layers):
                hidden_states = self.layer_compute(self.layers[idx], idx, hidden_states, position_ids, attention_mask, storage_ids)
        
        b, s, h = hidden_states.shape

        hidden_states = hidden_states.reshape(b * s, h)
        hidden_states = flashinfer.rmsnorm(hidden_states, self.norm_weight, self.norm_variance_epsilon)
        hidden_states = hidden_states.reshape(b, s, h)
        logits = F.linear(hidden_states, self.lm_head).float()
        return logits

    
def capture_graph(
    llm :LLM, decoding_seqlen :int =1, mempool=None, n_warmups :int=3
):
    device = llm.device
    dtype = llm.dtype
    bsz = llm.batch_size
    static_input_ids = torch.full((bsz, decoding_seqlen), 0, dtype=torch.long, device=device)
    static_position_ids = torch.full((bsz, decoding_seqlen), 0, dtype=torch.long, device=device)
    static_storage_ids = torch.arange(decoding_seqlen, dtype=torch.long, device=device)
    static_attn_mask = torch.full((decoding_seqlen, llm.max_length), 0, dtype=dtype, device=device)
    static_attn_mask = static_attn_mask[None, None, :, :]
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(n_warmups):
            static_logits = llm.inference(
                    input_ids=static_input_ids, 
                    position_ids=static_position_ids, 
                    attention_mask=static_attn_mask,
                    storage_ids=static_storage_ids, 
                    )
        s.synchronize()
    torch.cuda.current_stream().wait_stream(s)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=mempool):
        static_logits = llm.inference(
                input_ids=static_input_ids,  
                position_ids=static_position_ids, 
                attention_mask=static_attn_mask,
                storage_ids=static_storage_ids,
                )
    def run(input_ids, storage_ids, position_ids, attention_mask):
        static_input_ids.copy_(input_ids)
        static_storage_ids.copy_(storage_ids)
        static_position_ids.copy_(position_ids)
        static_attn_mask.copy_(attention_mask)
        graph.replay()
        return static_logits.clone()
    
    return run


class LLMEngine:
    def __init__(self, 
                model_name: str,
                batch_size :int = 1,
                max_length :int = 256, 
                device :str = 'cuda:0',
                dtype = torch.float16) -> None:
        
        self.llm = LLM(model_name, batch_size, max_length, device, dtype)
        self.callables = {}
        self.mempool = None

    @torch.inference_mode()
    def initialize_cuda_graph(self, 
            decoding_seqlens :List[int],
            n_warmups=12):
        gc.collect()
        self.mempool = torch.cuda.graphs.graph_pool_handle()
        for decoding_seqlen in decoding_seqlens:
            if decoding_seqlen not in self.callables:
                self.callables[decoding_seqlen] = capture_graph(
                    llm=self.llm,
                    decoding_seqlen=decoding_seqlen,
                    mempool=self.mempool,
                    n_warmups=n_warmups
                )
        self.llm.kv_cache.clear()

    @torch.inference_mode()
    def inference(self,
            input_ids: torch.LongTensor, 
            storage_ids :torch.LongTensor,
            position_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            ):
            dec_length = input_ids.shape[1]
            if dec_length in self.callables.keys():
                logits = self.callables[dec_length](input_ids, storage_ids, position_ids, attention_mask)
            else:
                logits = self.llm.inference(input_ids, position_ids, attention_mask, storage_ids)
            return logits

    @torch.inference_mode()
    def prefill(self,
            input_ids: torch.LongTensor, 
            storage_ids :torch.LongTensor,
            position_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            ):
           
            logits = self.llm.inference(input_ids, position_ids, attention_mask, storage_ids)
            return logits
