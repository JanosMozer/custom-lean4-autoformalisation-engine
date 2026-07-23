import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from transformers.activations import ACT2FN
import sys
import gc

# 1. Define CustomExpert and CustomQwen3MoeExperts so experts are individual nn.Linear layers
class CustomExpert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.moe_intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.moe_intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.moe_intermediate_size, config.hidden_size, bias=False)

class CustomQwen3MoeExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.act_fn = ACT2FN[config.hidden_act]
        
        # Register each expert as a submodule with string key "0", "1", ...
        for j in range(self.num_experts):
            self.add_module(str(j), CustomExpert(config))

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0].item()
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            
            expert = getattr(self, str(expert_idx))
            gate = expert.gate_proj(current_state)
            up = expert.up_proj(current_state)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = expert.down_proj(current_hidden_states)
            
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states

# 2. Monkey-patch Qwen3MoeExperts and _init_weights in transformers BEFORE importing
import transformers.models.qwen3_moe.modeling_qwen3_moe as modeling_qwen3_moe
modeling_qwen3_moe.Qwen3MoeExperts = CustomQwen3MoeExperts
modeling_qwen3_moe.Qwen3MoePreTrainedModel._init_weights = lambda self, module: None

print("Loading config...")
config = AutoConfig.from_pretrained('models/base_qwen', trust_remote_code=True)
config._attn_implementation = "flash_attention_2"

print("Defining BitsAndBytesConfig...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

print("Loading model and tokenizer on CPU...")
try:
    tokenizer = AutoTokenizer.from_pretrained('models/base_qwen', trust_remote_code=True)
    
    # Load model on CPU first (fits easily in 125GB CPU RAM)
    model = AutoModelForCausalLM.from_pretrained(
        'models/base_qwen',
        dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    print("Model loaded on CPU.")

    print("Replacing linear layers with 4-bit...")
    from transformers.integrations.bitsandbytes import replace_with_bnb_linear
    model = replace_with_bnb_linear(
        model,
        quantization_config=bnb_config,
        modules_to_not_convert=["lm_head"],
    )
    model.is_loaded_in_4bit = True
    model.quantization_method = "bitsandbytes"
    print("In-place 4-bit conversion complete.")

    print("Moving model to GPU layer-by-layer to prevent VRAM spikes/fragmentation...")
    model.model.embed_tokens = model.model.embed_tokens.to("cuda:0")
    model.model.norm = model.model.norm.to("cuda:0")
    model.lm_head = model.lm_head.to("cuda:0")
    
    for i in range(len(model.model.layers)):
        model.model.layers[i] = model.model.layers[i].to("cuda:0")
        gc.collect()
        torch.cuda.empty_cache()
        if (i + 1) % 8 == 0:
            print(f"  Moved {i + 1}/{len(model.model.layers)} layers to GPU...")

    # Clear CPU cache and print memory
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Total VRAM allocated by PyTorch: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # Freeze base model weights
    for param in model.parameters():
        param.requires_grad = False

    print("Running forward pass entirely on GPU...")
    dummy_input = torch.tensor([[100, 200, 300, 400]], dtype=torch.long, device="cuda:0")
    outputs = model(dummy_input)
    print("Forward pass successful!")
    print(f"Logits shape: {outputs.logits.shape}")

    print("Testing backward pass...")
    model.model.embed_tokens.weight.requires_grad = True
    loss = outputs.logits.sum()
    loss.backward()
    print("Backward pass successful!")
    print(f"Gradient of embed_tokens weight: {model.model.embed_tokens.weight.grad is not None}")

except Exception as e:
    import traceback
    print("Failed:")
    traceback.print_exc()
