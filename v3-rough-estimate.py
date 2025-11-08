#!/usr/bin/env python3
"""
DeepSeek-V3 FLOPs Profiler

Accurate FLOPs analysis tool for DeepSeek-V3
- Full modeling of low-rank Q/K/V projections
- Precise calculation of MoE routing and expert costs
- Distinguishes between Dense and MoE layers
- Supports Prefill and Decode mode analysis
"""

from typing import Dict, Tuple, Set
from transformers import AutoConfig


# ============================================================================
# Configuration - Modify parameters here
# ============================================================================
CONFIG = {
    "model_name": "deepseek-ai/DeepSeek-V3",
    
    # Analysis scenarios
    "batch_size": 4,
    "seq_len": 128,
    "gen_steps": 128,
    
    # Computation options
    "lm_last_token_only": True,      # Compute LM head only on last token
    "include_softmax_ops": True,     # Include Softmax scalar operations
    "include_rmsnorm_ops": True,     # Include RMSNorm scalar operations
    "include_pointwise_ops": True,   # Include SiLU/Gating pointwise operations
    "detailed_decode": False,        # Exact calculation for each decode step (slow but accurate)
    
    # Scalar cost tuning
    "softmax_scalar_cost": 5,        # Scalar operations per element for Softmax
    "silu_scalar_cost": 4,           # Scalar operations per element for SiLU
}
# ============================================================================


class DeepSeekV3FlopsCalculator:
    """DeepSeek-V3 FLOPs Calculator"""
    
    def __init__(self, config):
        # Basic parameters
        self.hidden_size = int(config.hidden_size)
        self.num_layers = int(config.num_hidden_layers)
        self.num_attention_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.vocab_size = int(config.vocab_size)
        
        # Low-rank attention
        self.q_lora_rank = int(config.q_lora_rank)
        self.kv_lora_rank = int(config.kv_lora_rank)
        self.qk_nope_head_dim = int(config.qk_nope_head_dim)
        self.qk_rope_head_dim = int(config.qk_rope_head_dim)
        self.v_head_dim = int(config.v_head_dim)
        
        # MoE parameters
        self.n_routed_experts = int(config.n_routed_experts)
        self.num_experts_per_tok = int(config.num_experts_per_tok)
        self.n_shared_experts = int(config.n_shared_experts)
        self.moe_intermediate_size = int(config.moe_intermediate_size)
        self.first_k_dense_replace = int(config.first_k_dense_replace)
        self.moe_layer_freq = int(config.moe_layer_freq)
        
        # Dense MLP
        self.intermediate_size = int(config.intermediate_size)
        
        # Router
        self.n_group = int(config.n_group)
        self.topk_group = int(config.topk_group)
        
        # Derived parameters
        self.q_per_head = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.q_out_dim = self.num_attention_heads * self.q_per_head
        self.k_out_dim = self.num_kv_heads * self.q_per_head
        self.v_out_dim = self.num_kv_heads * self.v_head_dim
        
        # Determine layer types
        self.dense_layers, self.moe_layers = self._determine_layer_types()
        
    def _determine_layer_types(self) -> Tuple[Set[int], Set[int]]:
        """Determine layer types (Dense MLP vs MoE)"""
        dense_layers = set(range(min(self.first_k_dense_replace, self.num_layers)))
        
        moe_layers = set()
        for i in range(self.num_layers):
            if i not in dense_layers:
                if (i - self.first_k_dense_replace) % self.moe_layer_freq == 0:
                    moe_layers.add(i)
        
        # Check coverage
        total_covered = len(dense_layers) + len(moe_layers)
        if total_covered < self.num_layers:
            skipped = self.num_layers - total_covered
            print(f"Warning: {skipped} layers are neither Dense nor MoE (moe_layer_freq={self.moe_layer_freq})")
        
        return dense_layers, moe_layers
    
    def calculate_flops(
        self,
        batch_size: int,
        seq_len: int,
        mode: str = "prefill",
        kv_cache_len: int = 0,
        lm_last_token_only: bool = True,
        include_softmax_ops: bool = True,
        include_rmsnorm_ops: bool = True,
        include_pointwise_ops: bool = True,
        softmax_scalar_cost: int = 5,
        silu_scalar_cost: int = 4,
    ) -> Tuple[float, Dict[str, float], Dict]:
        """
        Calculate FLOPs for a single forward pass
        
        Returns:
            (total_TFLOPs, flops_breakdown, metadata)
        """
        
        # Determine sequence lengths
        if mode == "prefill":
            process_seq_len = seq_len
            attn_seq_len = seq_len
            lm_tokens = batch_size if lm_last_token_only else batch_size * seq_len
        else:  # decode
            process_seq_len = 1
            attn_seq_len = kv_cache_len + 1
            lm_tokens = batch_size
        
        total_tokens = batch_size * process_seq_len
        
        # FLOPs counters
        flops = {
            "attention_qkv_proj": 0,
            "attention_compute": 0,
            "attention_output_proj": 0,
            "moe_router": 0,
            "moe_routed_experts": 0,
            "moe_shared_experts": 0,
            "dense_mlp": 0,
            "lm_head": 0,
            "rmsnorm": 0,
            "mlp_pointwise": 0,
        }
        
        # Per-layer calculation
        for layer_idx in range(self.num_layers):
            # ========== Attention Module ==========
            
            # Q projection (two-stage low-rank)
            # hidden -> q_lora_rank -> q_out
            flops["attention_qkv_proj"] += 2 * self.hidden_size * self.q_lora_rank * total_tokens
            flops["attention_qkv_proj"] += 2 * self.q_lora_rank * self.q_out_dim * total_tokens
            
            # K projection
            flops["attention_qkv_proj"] += 2 * self.hidden_size * self.kv_lora_rank * total_tokens
            flops["attention_qkv_proj"] += 2 * self.kv_lora_rank * self.k_out_dim * total_tokens
            
            # V projection
            flops["attention_qkv_proj"] += 2 * self.hidden_size * self.kv_lora_rank * total_tokens
            flops["attention_qkv_proj"] += 2 * self.kv_lora_rank * self.v_out_dim * total_tokens
            
            # Attention computation
            # QK^T: [B, H, N, D] @ [B, H, D, M] -> [B, H, N, M]
            qkt_flops = 2 * batch_size * self.num_attention_heads * process_seq_len * attn_seq_len * self.q_per_head
            
            # Softmax scalar operations
            softmax_flops = 0
            if include_softmax_ops:
                num_scores = batch_size * self.num_attention_heads * process_seq_len * attn_seq_len
                softmax_flops = softmax_scalar_cost * num_scores
            
            # AV: [B, H, N, M] @ [B, H, M, D_v] -> [B, H, N, D_v]
            av_flops = 2 * batch_size * self.num_attention_heads * process_seq_len * attn_seq_len * self.v_head_dim
            
            flops["attention_compute"] += qkt_flops + softmax_flops + av_flops
            
            # O projection (dense)
            flops["attention_output_proj"] += 2 * self.v_out_dim * self.hidden_size * total_tokens
            
            # RMSNorm (2 per layer)
            if include_rmsnorm_ops:
                # ~4 ops per element: mean(x^2), sqrt, div, scale
                flops["rmsnorm"] += 2 * 4 * self.hidden_size * total_tokens
            
            # ========== MLP/MoE Module ==========
            
            if layer_idx in self.dense_layers:
                # Dense Gated FFN: up + gate + down
                flops["dense_mlp"] += 2 * self.hidden_size * self.intermediate_size * total_tokens  # up
                flops["dense_mlp"] += 2 * self.hidden_size * self.intermediate_size * total_tokens  # gate
                flops["dense_mlp"] += 2 * self.intermediate_size * self.hidden_size * total_tokens  # down
                
                if include_pointwise_ops:
                    # SiLU(up) * gate
                    flops["mlp_pointwise"] += (silu_scalar_cost + 1) * self.intermediate_size * total_tokens
                    
            elif layer_idx in self.moe_layers:
                # Router (gate2expert)
                router_proj = 2 * self.hidden_size * self.n_routed_experts * total_tokens
                router_softmax = softmax_scalar_cost * batch_size * process_seq_len * self.n_routed_experts
                
                experts_per_group = self.n_routed_experts // self.n_group
                router_topk = self.num_experts_per_tok * batch_size * process_seq_len * experts_per_group
                
                flops["moe_router"] += router_proj + router_softmax + router_topk
                
                # Expert FFN (same structure as dense FFN)
                expert_up = 2 * self.hidden_size * self.moe_intermediate_size
                expert_gate = 2 * self.hidden_size * self.moe_intermediate_size
                expert_down = 2 * self.moe_intermediate_size * self.hidden_size
                expert_total = expert_up + expert_gate + expert_down
                
                # Routed experts (num_experts_per_tok active per token)
                flops["moe_routed_experts"] += expert_total * self.num_experts_per_tok * total_tokens
                
                # Shared experts (always active)
                if self.n_shared_experts > 0:
                    flops["moe_shared_experts"] += expert_total * self.n_shared_experts * total_tokens
                
                if include_pointwise_ops:
                    total_active_experts = self.num_experts_per_tok + self.n_shared_experts
                    flops["mlp_pointwise"] += (silu_scalar_cost + 1) * self.moe_intermediate_size * total_active_experts * total_tokens
        
        # LM Head
        flops["lm_head"] = 2 * self.hidden_size * self.vocab_size * lm_tokens
        
        # Total
        total_flops = sum(flops.values())
        total_tflops = total_flops / 1e12
        
        # Metadata
        meta = {
            "mode": mode,
            "batch_size": batch_size,
            "process_seq_len": process_seq_len,
            "attn_seq_len": attn_seq_len,
            "total_tokens": total_tokens,
            "lm_tokens": lm_tokens,
        }
        
        return total_tflops, flops, meta


def print_breakdown(title: str, flops_dict: Dict[str, float], total_tokens: int):
    """Print FLOPs breakdown table"""
    total = sum(flops_dict.values())
    
    print(f"\n{'='*75}")
    print(f"{title}")
    print(f"{'='*75}")
    print(f"{'Component':<35} {'TFLOPs':<12} {'%':<8} {'GFLOPs/tok':<12}")
    print("-"*75)
    
    for name, fl in sorted(flops_dict.items(), key=lambda x: x[1], reverse=True):
        if fl <= 0:
            continue
        pct = (fl / total * 100) if total > 0 else 0.0
        gflops_per_tok = fl / max(total_tokens, 1) / 1e9
        tflops = fl / 1e12
        print(f"{name:<35} {tflops:>10.3f}   {pct:>6.2f}%   {gflops_per_tok:>10.3f}")
    
    print("-"*75)
    total_gflops_per_tok = total / max(total_tokens, 1) / 1e9
    print(f"{'TOTAL':<35} {total/1e12:>10.3f}   100.00%   {total_gflops_per_tok:>10.3f}")
    print("="*75)


def main():
    cfg = CONFIG
    
    print(f"\nLoading config from {cfg['model_name']}...")
    config = AutoConfig.from_pretrained(cfg["model_name"], trust_remote_code=True)
    
    calculator = DeepSeekV3FlopsCalculator(config)
    
    # Architecture summary
    print(f"\nDeepSeek-V3 Architecture Summary:")
    print(f"   Layers: {calculator.num_layers} ({len(calculator.dense_layers)} Dense + {len(calculator.moe_layers)} MoE)")
    print(f"   Hidden size: {calculator.hidden_size}")
    print(f"   Attention heads: {calculator.num_attention_heads} (Q), {calculator.num_kv_heads} (KV)")
    print(f"   Head dims: {calculator.q_per_head} (Q/K), {calculator.v_head_dim} (V)")
    print(f"   Q/K/V low-rank: {calculator.q_lora_rank} / {calculator.kv_lora_rank} / {calculator.kv_lora_rank}")
    print(f"   MoE config: {calculator.n_routed_experts} routed experts, top-{calculator.num_experts_per_tok}, {calculator.n_shared_experts} shared")
    print(f"   Vocab size: {calculator.vocab_size:,}")
    
    common_kwargs = {
        "lm_last_token_only": cfg["lm_last_token_only"],
        "include_softmax_ops": cfg["include_softmax_ops"],
        "include_rmsnorm_ops": cfg["include_rmsnorm_ops"],
        "include_pointwise_ops": cfg["include_pointwise_ops"],
        "softmax_scalar_cost": cfg["softmax_scalar_cost"],
        "silu_scalar_cost": cfg["silu_scalar_cost"],
    }
    
    # === Prefill Analysis ===
    print(f"\n{'='*75}")
    print("Analyzing PREFILL phase...")
    print(f"{'='*75}")
    
    prefill_tflops, prefill_flops, prefill_meta = calculator.calculate_flops(
        batch_size=cfg["batch_size"],
        seq_len=cfg["seq_len"],
        mode="prefill",
        **common_kwargs
    )
    
    print_breakdown(
        f"PREFILL (B={cfg['batch_size']}, L={cfg['seq_len']})",
        prefill_flops,
        prefill_meta["total_tokens"]
    )
    
    # === Decode First Step Analysis ===
    print(f"\n{'='*75}")
    print("Analyzing DECODE phase (first step)...")
    print(f"{'='*75}")
    
    dec1_tflops, dec1_flops, dec1_meta = calculator.calculate_flops(
        batch_size=cfg["batch_size"],
        seq_len=1,
        mode="decode",
        kv_cache_len=cfg["seq_len"],
        **common_kwargs
    )
    
    print_breakdown(
        f"DECODE Step 1 (B={cfg['batch_size']}, KV={cfg['seq_len']})",
        dec1_flops,
        dec1_meta["total_tokens"]
    )
    
    # === Decode Last Step Analysis ===
    print(f"\n{'='*75}")
    print(f"Analyzing DECODE phase (step {cfg['gen_steps']})...")
    print(f"{'='*75}")
    
    last_kv_len = cfg["seq_len"] + cfg["gen_steps"] - 1
    decN_tflops, decN_flops, decN_meta = calculator.calculate_flops(
        batch_size=cfg["batch_size"],
        seq_len=1,
        mode="decode",
        kv_cache_len=last_kv_len,
        **common_kwargs
    )
    
    print_breakdown(
        f"DECODE Step {cfg['gen_steps']} (B={cfg['batch_size']}, KV={last_kv_len})",
        decN_flops,
        decN_meta["total_tokens"]
    )
    
    # === Total Decode ===
    print(f"\n{'='*75}")
    print(f"Computing TOTAL DECODE FLOPs for {cfg['gen_steps']} steps...")
    print(f"{'='*75}")
    
    if cfg["detailed_decode"]:
        print("Exact calculation (may take a moment)...")
        total_decode_tflops = 0.0
        step_list = []
        for step in range(cfg["gen_steps"]):
            kv_len = cfg["seq_len"] + step
            tflops, _, _ = calculator.calculate_flops(
                batch_size=cfg["batch_size"],
                seq_len=1,
                mode="decode",
                kv_cache_len=kv_len,
                **common_kwargs
            )
            step_list.append(tflops)
            total_decode_tflops += tflops
        
        print(f"Exact total: {total_decode_tflops:.2f} TFLOPs")
        print(f"   Step 1:      {step_list[0]:.4f} TFLOPs")
        print(f"   Step {cfg['gen_steps']}:     {step_list[-1]:.4f} TFLOPs")
        print(f"   Average:     {total_decode_tflops/cfg['gen_steps']:.4f} TFLOPs/step")
    else:
        # Linear interpolation estimation
        avg_tflops = 0.5 * (dec1_tflops + decN_tflops)
        total_decode_tflops = avg_tflops * cfg["gen_steps"]
        print(f"Linear interpolation:")
        print(f"   Step 1:      {dec1_tflops:.4f} TFLOPs")
        print(f"   Step {cfg['gen_steps']}:     {decN_tflops:.4f} TFLOPs")
        print(f"   Average:     {avg_tflops:.4f} TFLOPs/step")
        print(f"   Total:       {total_decode_tflops:.2f} TFLOPs")
    
    # === Key Insights ===
    print(f"\n{'='*75}")
    print("KEY INSIGHTS")
    print(f"{'='*75}")
    
    prefill_total = sum(prefill_flops.values())
    dec1_total = sum(dec1_flops.values())
    decN_total = sum(decN_flops.values())
    
    # 1. MoE contribution analysis
    moe_router = prefill_flops.get("moe_router", 0)
    moe_routed = prefill_flops.get("moe_routed_experts", 0)
    moe_shared = prefill_flops.get("moe_shared_experts", 0)
    moe_total = moe_router + moe_routed + moe_shared
    
    print(f"\n1. MoE Contribution (Prefill):")
    print(f"   Router:         {moe_router/1e12:>8.3f} TFLOPs ({moe_router/prefill_total*100:>5.2f}%)")
    print(f"   Routed experts: {moe_routed/1e12:>8.3f} TFLOPs ({moe_routed/prefill_total*100:>5.2f}%)")
    print(f"   Shared experts: {moe_shared/1e12:>8.3f} TFLOPs ({moe_shared/prefill_total*100:>5.2f}%)")
    print(f"   {'─'*31}")
    print(f"   MoE Total:      {moe_total/1e12:>8.3f} TFLOPs ({moe_total/prefill_total*100:>5.2f}%)")
    
    # 2. Attention compute growth
    attn_prefill = prefill_flops["attention_compute"]
    attn_dec1 = dec1_flops["attention_compute"]
    attn_decN = decN_flops["attention_compute"]
    attn_growth = attn_decN / max(attn_dec1, 1)
    
    print(f"\n2. Attention Compute Scaling:")
    print(f"   Prefill (L={cfg['seq_len']}):              {attn_prefill/1e12:>8.3f} TFLOPs ({attn_prefill/prefill_total*100:>5.2f}%)")
    print(f"   Decode step 1 (KV={cfg['seq_len']}):      {attn_dec1/1e12:>8.3f} TFLOPs ({attn_dec1/dec1_total*100:>5.2f}%)")
    print(f"   Decode step {cfg['gen_steps']} (KV={last_kv_len}): {attn_decN/1e12:>8.3f} TFLOPs ({attn_decN/decN_total*100:>5.2f}%)")
    print(f"   Growth: {attn_growth:.2f}x (decode step 1 -> {cfg['gen_steps']})")
    
    # 3. Per-token FLOPs
    prefill_per_tok = prefill_total / prefill_meta["total_tokens"] / 1e9
    dec1_per_tok = dec1_total / dec1_meta["total_tokens"] / 1e9
    decN_per_tok = decN_total / decN_meta["total_tokens"] / 1e9
    
    print(f"\n3. Per-Token FLOPs:")
    print(f"   Prefill:        {prefill_per_tok:>8.2f} GFLOPs/token")
    print(f"   Decode step 1:  {dec1_per_tok:>8.2f} GFLOPs/token")
    print(f"   Decode step {cfg['gen_steps']}: {decN_per_tok:>8.2f} GFLOPs/token")
    print(f"   Growth: {decN_per_tok/dec1_per_tok:.2f}x (decode only)")
    
    # 4. Overall FLOPs
    total_flops = prefill_tflops + total_decode_tflops
    
    print(f"\n4. Total FLOPs Summary:")
    print(f"   Prefill ({cfg['seq_len']} tokens):     {prefill_tflops:>10.2f} TFLOPs ({prefill_tflops/total_flops*100:>5.2f}%)")
    print(f"   Decode ({cfg['gen_steps']} steps):      {total_decode_tflops:>10.2f} TFLOPs ({total_decode_tflops/total_flops*100:>5.2f}%)")
    print(f"   {'─'*43}")
    print(f"   Grand Total:              {total_flops:>10.2f} TFLOPs")
    
    print(f"\n{'='*75}\n")


if __name__ == "__main__":
    main()