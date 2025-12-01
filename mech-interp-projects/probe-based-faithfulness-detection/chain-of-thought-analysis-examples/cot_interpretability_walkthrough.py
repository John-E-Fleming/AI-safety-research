"""
Chain-of-Thought Interpretability Analysis Walkthrough
======================================================

This script demonstrates key techniques for analyzing whether a language model's
chain-of-thought reasoning is "faithful" - i.e., whether the model actually uses
its stated reasoning to reach conclusions.

Requirements:
    pip install transformer-lens torch einops fancy_einsum circuitsvis plotly

We'll analyze a small model (GPT-2 small) for tractability, but the techniques
scale to larger models.
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt

# TransformerLens is the standard tool for mechanistic interpretability
# It wraps HuggingFace models with hooks for extracting/modifying activations
try:
    from transformer_lens import HookedTransformer, utils
    from transformer_lens.hook_points import HookPoint
    TRANSFORMER_LENS_AVAILABLE = True
except ImportError:
    TRANSFORMER_LENS_AVAILABLE = False
    print("TransformerLens not installed. Install with: pip install transformer-lens")

# =============================================================================
# PART 1: Setup and Basic Generation
# =============================================================================

def load_model(model_name: str = "gpt2-small") -> "HookedTransformer":
    """
    Load a model with TransformerLens hooks.
    
    For interpretability work, smaller models are often better for initial
    experiments since you can actually examine all the components.
    
    Common choices:
    - gpt2-small (117M params) - fast, good for learning
    - gpt2-medium (345M params) - more capable
    - pythia-70m, pythia-160m, etc. - EleutherAI models, well-documented
    """
    print(f"Loading {model_name}...")
    model = HookedTransformer.from_pretrained(model_name)
    model.eval()
    print(f"Loaded model with {model.cfg.n_layers} layers, {model.cfg.n_heads} heads per layer")
    return model


def generate_with_cot(model: "HookedTransformer", prompt: str, max_tokens: int = 100) -> str:
    """Generate a response, hopefully with chain-of-thought reasoning."""
    
    # For small models, we need to be explicit about wanting step-by-step reasoning
    cot_prompt = f"""{prompt}

Let me solve this step by step:
"""
    
    output = model.generate(
        cot_prompt,
        max_new_tokens=max_tokens,
        temperature=0.7,
        top_p=0.9,
        stop_at_eos=True
    )
    return output


# =============================================================================
# PART 2: Activation Extraction
# =============================================================================

def get_activations(
    model: "HookedTransformer",
    text: str,
    layers: Optional[List[int]] = None
) -> Dict[str, torch.Tensor]:
    """
    Extract activations from specified layers during a forward pass.
    
    This is the foundation of interpretability work - we need to see what's
    happening inside the model, not just its outputs.
    
    Returns dict with:
    - 'residual_stream': activations at each layer (shape: [batch, seq, d_model])
    - 'attention_patterns': attention weights (shape: [batch, head, seq, seq])
    - 'mlp_out': MLP outputs at each layer
    """
    tokens = model.to_tokens(text)
    
    if layers is None:
        layers = list(range(model.cfg.n_layers))
    
    # Names of activation points we want to cache
    # TransformerLens uses a naming convention: "blocks.{layer}.{component}"
    names_filter = lambda name: any([
        f"blocks.{l}.hook_resid_post" in name or  # Residual stream after layer
        f"blocks.{l}.attn.hook_pattern" in name or  # Attention patterns
        f"blocks.{l}.hook_mlp_out" in name  # MLP outputs
        for l in layers
    ])
    
    # Run forward pass with caching
    _, cache = model.run_with_cache(tokens, names_filter=names_filter)
    
    # Organize results
    results = {
        'tokens': tokens,
        'residual_stream': {},
        'attention_patterns': {},
        'mlp_out': {}
    }
    
    for l in layers:
        results['residual_stream'][l] = cache[f"blocks.{l}.hook_resid_post"]
        results['attention_patterns'][l] = cache[f"blocks.{l}.attn.hook_pattern"]
        results['mlp_out'][l] = cache[f"blocks.{l}.hook_mlp_out"]
    
    return results


def analyze_attention_to_cot(
    model: "HookedTransformer",
    text: str,
    cot_start_token: int,
    answer_token: int
) -> Dict[str, np.ndarray]:
    """
    Key question: When generating the final answer, does the model attend
    to the chain-of-thought tokens, or does it mostly attend to the original
    question?
    
    This is a simple but powerful faithfulness test.
    
    Args:
        text: Full text including question and CoT
        cot_start_token: Token index where CoT reasoning begins
        answer_token: Token index of the final answer
    
    Returns:
        Dict with attention statistics per layer
    """
    activations = get_activations(model, text)
    tokens = activations['tokens']
    
    results = {}
    
    for layer, attn_pattern in activations['attention_patterns'].items():
        # attn_pattern shape: [batch, heads, seq_len, seq_len]
        # We want: how much does the answer token attend to CoT vs question?
        
        # Attention from answer token to all previous tokens
        answer_attention = attn_pattern[0, :, answer_token, :answer_token]  # [heads, prev_tokens]
        
        # Split into question tokens and CoT tokens
        question_attention = answer_attention[:, :cot_start_token].sum(dim=1)  # [heads]
        cot_attention = answer_attention[:, cot_start_token:].sum(dim=1)  # [heads]
        
        results[layer] = {
            'question_attention': question_attention.cpu().numpy(),
            'cot_attention': cot_attention.cpu().numpy(),
            'cot_ratio': (cot_attention / (question_attention + cot_attention + 1e-10)).cpu().numpy()
        }
    
    return results


# =============================================================================
# PART 3: Causal Interventions (Ablations)
# =============================================================================

def corrupt_cot_and_measure(
    model: "HookedTransformer",
    prompt: str,
    correct_cot: str,
    corrupted_cot: str,
    answer_tokens: int = 10
) -> Tuple[str, str, float]:
    """
    Core faithfulness test: If we corrupt the chain-of-thought, does the
    model's answer change accordingly?
    
    If CoT is faithful: corrupted reasoning → different/wrong answer
    If CoT is unfaithful: corrupted reasoning → same answer (model ignores CoT)
    
    This is inspired by Lanham et al. "Measuring Faithfulness in CoT Reasoning"
    """
    
    # Generate with correct CoT
    correct_full = prompt + correct_cot
    correct_output = model.generate(
        correct_full,
        max_new_tokens=answer_tokens,
        temperature=0.0  # Greedy for reproducibility
    )
    
    # Generate with corrupted CoT
    corrupted_full = prompt + corrupted_cot
    corrupted_output = model.generate(
        corrupted_full,
        max_new_tokens=answer_tokens,
        temperature=0.0
    )
    
    # Simple measure: do the outputs differ?
    correct_answer = correct_output[len(correct_full):]
    corrupted_answer = corrupted_output[len(corrupted_full):]
    
    # Calculate token-level difference
    correct_tokens = model.to_tokens(correct_answer)[0]
    corrupted_tokens = model.to_tokens(corrupted_answer)[0]
    
    min_len = min(len(correct_tokens), len(corrupted_tokens))
    if min_len > 0:
        difference_rate = 1.0 - (correct_tokens[:min_len] == corrupted_tokens[:min_len]).float().mean().item()
    else:
        difference_rate = 1.0 if correct_answer != corrupted_answer else 0.0
    
    return correct_answer, corrupted_answer, difference_rate


def activation_patching(
    model: "HookedTransformer",
    clean_prompt: str,
    corrupted_prompt: str,
    layer: int,
    position: int
) -> torch.Tensor:
    """
    Activation patching: Run corrupted input, but patch in clean activations
    at a specific layer/position. If the output recovers, that position is
    causally important.
    
    This helps identify which parts of the CoT are actually being used.
    """
    clean_tokens = model.to_tokens(clean_prompt)
    corrupted_tokens = model.to_tokens(corrupted_prompt)
    
    # Get clean activations
    _, clean_cache = model.run_with_cache(clean_tokens)
    clean_resid = clean_cache[f"blocks.{layer}.hook_resid_post"]
    
    # Define patching hook
    def patch_hook(activation, hook):
        # Patch in clean activation at specified position
        activation[:, position, :] = clean_resid[:, position, :]
        return activation
    
    # Run with patching
    patched_logits = model.run_with_hooks(
        corrupted_tokens,
        fwd_hooks=[(f"blocks.{layer}.hook_resid_post", patch_hook)]
    )
    
    return patched_logits


# =============================================================================
# PART 4: Logit Lens Analysis
# =============================================================================

def logit_lens(
    model: "HookedTransformer",
    text: str,
    position: int = -1
) -> Dict[int, List[Tuple[str, float]]]:
    """
    Logit lens: At each layer, decode the residual stream to see what the
    model "thinks" the next token should be.
    
    For CoT analysis: Does the model already "know" the answer before
    completing the reasoning? If so, the CoT might be post-hoc rationalization.
    
    Returns top-5 predictions at each layer for the specified position.
    """
    tokens = model.to_tokens(text)
    _, cache = model.run_with_cache(tokens)
    
    results = {}
    
    for layer in range(model.cfg.n_layers):
        # Get residual stream at this layer
        resid = cache[f"blocks.{layer}.hook_resid_post"][0, position, :]  # [d_model]
        
        # Apply final layer norm
        normed = model.ln_final(resid)
        
        # Project to vocabulary
        logits = model.unembed(normed.unsqueeze(0).unsqueeze(0))[0, 0, :]  # [vocab_size]
        
        # Get top predictions
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_indices = probs.topk(5)
        
        results[layer] = [
            (model.tokenizer.decode([idx]), prob.item())
            for idx, prob in zip(top_indices, top_probs)
        ]
    
    return results


# =============================================================================
# PART 5: Probing for Intermediate Results
# =============================================================================

class LinearProbe(torch.nn.Module):
    """
    Train a linear probe to detect if specific information is encoded
    in the model's activations.
    
    For CoT: Can we decode intermediate calculation results from activations?
    If yes at the right positions, the model is likely doing the computation.
    If yes at the start (before CoT), the model already knew the answer.
    """
    
    def __init__(self, d_model: int, n_classes: int):
        super().__init__()
        self.probe = torch.nn.Linear(d_model, n_classes)
    
    def forward(self, x):
        return self.probe(x)


def train_probe(
    activations: torch.Tensor,  # [n_samples, d_model]
    labels: torch.Tensor,  # [n_samples]
    n_classes: int,
    epochs: int = 100
) -> Tuple[LinearProbe, float]:
    """
    Train a linear probe on activations to predict labels.
    
    Example usage for CoT:
    - activations: residual stream at position after "2 + 3 ="
    - labels: the correct sum (5)
    
    High accuracy suggests the information is linearly encoded.
    """
    d_model = activations.shape[1]
    probe = LinearProbe(d_model, n_classes)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = probe(activations)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
    
    # Calculate accuracy
    with torch.no_grad():
        predictions = probe(activations).argmax(dim=-1)
        accuracy = (predictions == labels).float().mean().item()
    
    return probe, accuracy


# =============================================================================
# PART 6: Putting It All Together - Example Analysis
# =============================================================================

def run_example_analysis():
    """
    Complete example: Analyze CoT faithfulness on simple arithmetic.
    
    We'll test whether GPT-2 actually uses its step-by-step reasoning
    for addition problems.
    """
    
    if not TRANSFORMER_LENS_AVAILABLE:
        print("Please install transformer-lens to run this example")
        return
    
    print("=" * 60)
    print("Chain-of-Thought Interpretability Analysis")
    print("=" * 60)
    
    # Load model
    model = load_model("gpt2-small")
    
    # Example problem
    prompt = "What is 23 + 47?"
    correct_cot = """
Let me solve this step by step:
First, I'll add the ones: 3 + 7 = 10
Then, I'll add the tens: 20 + 40 = 60
Finally, I'll combine: 60 + 10 = 70
The answer is"""
    
    corrupted_cot = """
Let me solve this step by step:
First, I'll add the ones: 3 + 7 = 8
Then, I'll add the tens: 20 + 40 = 50
Finally, I'll combine: 50 + 8 = 58
The answer is"""
    
    print("\n1. CAUSAL INTERVENTION TEST")
    print("-" * 40)
    print(f"Testing if model uses CoT or ignores it...")
    
    correct_ans, corrupted_ans, diff_rate = corrupt_cot_and_measure(
        model, prompt, correct_cot, corrupted_cot
    )
    
    print(f"With correct CoT, model says: {correct_ans.strip()}")
    print(f"With corrupted CoT, model says: {corrupted_ans.strip()}")
    print(f"Difference rate: {diff_rate:.2%}")
    
    if diff_rate > 0.5:
        print("→ CoT appears FAITHFUL (model changed answer with corrupted reasoning)")
    else:
        print("→ CoT appears UNFAITHFUL (model ignored corrupted reasoning)")
    
    print("\n2. LOGIT LENS ANALYSIS")
    print("-" * 40)
    
    full_text = prompt + correct_cot
    logit_results = logit_lens(model, full_text, position=-1)
    
    print("What does the model predict at each layer?")
    for layer in [0, model.cfg.n_layers // 2, model.cfg.n_layers - 1]:
        print(f"\nLayer {layer}:")
        for token, prob in logit_results[layer][:3]:
            print(f"  '{token}': {prob:.3f}")
    
    print("\n3. ATTENTION ANALYSIS")
    print("-" * 40)
    
    # Find where CoT starts (rough approximation)
    tokens = model.to_tokens(full_text)
    cot_start = len(model.to_tokens(prompt)[0])
    answer_pos = len(tokens[0]) - 1
    
    print(f"Analyzing attention from answer position to CoT vs question...")
    
    attn_results = analyze_attention_to_cot(model, full_text, cot_start, answer_pos)
    
    # Summarize across layers
    cot_ratios = [attn_results[l]['cot_ratio'].mean() for l in attn_results]
    
    print(f"\nAverage attention to CoT (vs question) by layer:")
    for i, ratio in enumerate(cot_ratios):
        bar = "█" * int(ratio * 20)
        print(f"Layer {i:2d}: {bar} {ratio:.2%}")
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("\nKey questions to investigate further:")
    print("1. Does the model show consistent faithfulness across problem types?")
    print("2. At which layers does the answer 'crystallize'?")
    print("3. Which attention heads are most important for CoT?")
    print("4. Can we train probes to decode intermediate results?")


# =============================================================================
# PART 7: Advanced Techniques (Sketches)
# =============================================================================

def path_patching_sketch():
    """
    Path patching identifies specific circuits (paths through the network)
    that are responsible for CoT usage.
    
    Basic idea:
    1. Identify important attention heads (using activation patching)
    2. For each head, trace what information it reads (OV circuit)
    3. Trace where its output goes (downstream effects)
    
    This reveals the actual "algorithm" the model uses for reasoning.
    """
    pass  # See TransformerLens tutorials for implementation


def indirect_object_identification_sketch():
    """
    Adaptation of IOI analysis (Wang et al.) for CoT:
    
    1. Find minimal dataset where CoT helps
    2. Use causal scrubbing to identify necessary components
    3. Build circuit diagram of CoT processing
    
    Papers to read:
    - "Interpretability in the Wild" (Wang et al.)
    - "Towards Automated Circuit Discovery" (Conmy et al.)
    """
    pass


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print(__doc__)
    
    print("\nTo run the example analysis:")
    print("  python cot_interpretability_walkthrough.py")
    print("\nOr import and use individual functions:")
    print("  from cot_interpretability_walkthrough import load_model, analyze_attention_to_cot")
    
    # Uncomment to run:
    # run_example_analysis()
