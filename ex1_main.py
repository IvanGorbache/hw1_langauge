import matplotlib
matplotlib.use('TkAgg')  # Use a standard windowing backend
import matplotlib.pyplot as plt
import torch
import pandas as pd
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os

# --- Configuration ---
MODEL_NAME = "gpt2"  # GPT-2 Small (124M)
DATA_PATH = "ex1_data.csv"
OUTPUT_CSV = "results.csv"


def get_subject_token_index(prompt, subject, tokenizer):
    """Finds the index of the last token of the subject in the tokenized prompt."""
    inputs = tokenizer(prompt, return_tensors="pt")
    prompt_ids = inputs['input_ids'][0].tolist()
    prompt_tokens = [tokenizer.decode([tid]) for tid in prompt_ids]

    # Subject might be multiple tokens
    subject_ids = tokenizer.encode(subject, add_prefix_space=True)
    subject_tokens = [tokenizer.decode([tid]) for tid in subject_ids]

    # Simple search for the subject tokens in the prompt
    n = len(subject_ids)
    for i in range(len(prompt_ids) - n, -1, -1):
        if prompt_ids[i:i + n] == subject_ids:
            return i + n - 1  # Index of the last token of the subject

    # Heuristic fallback if prefix space causes mismatch
    subject_plain = subject.strip().lower()
    for i in range(len(prompt_tokens) - 1, -1, -1):
        if subject_plain in prompt_tokens[i].strip().lower():
            return i
    return len(prompt_ids) - 2  # Default to one before last if not found


class HeadAblator:
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.n_heads = model.config.n_head
        self.head_dim = model.config.n_embd // self.n_heads

    def ablation_hook(self, head_indices):
        def hook(module, input):  # Removed 'output'
            modified_input = input[0].clone()
            for h_idx in head_indices:
                start = h_idx * self.head_dim
                end = (h_idx + 1) * self.head_dim
                modified_input[:, :, start:end] = 0
            return (modified_input,)

        return hook

    def apply_ablation(self, layer_head_dict):
        self.remove_hooks()
        for layer_idx, heads in layer_head_dict.items():
            # GPT2 Attention output projection
            target_module = self.model.transformer.h[layer_idx].attn.c_proj
            # Use register_forward_pre_hook to catch it before projection
            hook = target_module.register_forward_pre_hook(self.ablation_hook(heads))
            self.hooks.append(hook)

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def run_experiment():
    # Load model & tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME, output_attentions=True)
    model.eval()
    ablator = HeadAblator(model)
    n_heads = model.config.n_head

    df = pd.read_csv(DATA_PATH)
    results = []

    for idx, row in df.iterrows():
        prompt = row['Prompt']
        subject = row['Subject Word(s)']
        target_token_str = " " + row['Target Token'].strip()  # GPT-2 tokens often have leading space
        target_token_id = tokenizer.encode(target_token_str)[0]

        # 1. Baseline Run
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)

        last_logit = outputs.logits[0, -1, :]
        baseline_prob = torch.softmax(last_logit, dim=-1)[target_token_id].item()

        # Find Subject Index
        subj_idx = get_subject_token_index(prompt, subject, tokenizer)

        # 2. Map Attention Weights
        # attentions: tuple of (batch, heads, seq, seq)
        all_attentions = torch.stack(outputs.attentions)  # (layers, batch, heads, seq, seq)
        # Weight from last token (-1) to subject token (subj_idx)
        weights = all_attentions[:, 0, :, -1, subj_idx]  # (layers, heads)

        # Flatten to find top heads
        flat_weights = weights.flatten()
        top_indices = torch.argsort(flat_weights, descending=True)

        # Condition A: Single strongest head
        best_flat_idx = top_indices[0].item()
        best_layer, best_head = divmod(best_flat_idx, n_heads)

        # Condition B: Top 3 heads
        top3_flat = top_indices[:3].tolist()
        cond_b_heads = {}
        for f_idx in top3_flat:
            l, h = divmod(f_idx, n_heads)
            cond_b_heads.setdefault(l, []).append(h)

        # Condition C: All heads > 0.25
        cond_c_flat = torch.where(flat_weights > 0.25)[0].tolist()
        cond_c_heads = {}
        for f_idx in cond_c_flat:
            l, h = divmod(f_idx, n_heads)
            cond_c_heads.setdefault(l, []).append(h)

        # Function to measure delta
        def measure_intervention(heads_dict):
            if not heads_dict: return 0.0
            ablator.apply_ablation(heads_dict)
            with torch.no_grad():
                out = model(**inputs)
            ablator.remove_hooks()
            p_int = torch.softmax(out.logits[0, -1, :], dim=-1)[target_token_id].item()
            return (baseline_prob - p_int) / baseline_prob

        # Collect Deltas
        delta_a = measure_intervention({best_layer: [best_head]})
        delta_b = measure_intervention(cond_b_heads)
        delta_c = measure_intervention(cond_c_heads)

        results.append({
            'prompt_id': idx,
            'baseline_prob': baseline_prob,
            'condition_a_delta': delta_a,
            'condition_b_delta': delta_b,
            'condition_c_delta': delta_c,
            'max_head_layer': best_layer,
            'max_head_index': best_head
        })
        print(f"Processed prompt {idx}: {prompt[:30]}...")

    # Save results
    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
    print(f"Results saved to {OUTPUT_CSV}")


def get_top_k_tokens(logits, tokenizer, k=5):
    """Returns top k tokens and their probabilities for the visualization."""
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, k)

    # Corrected iteration for PyTorch tensors
    top_tokens = [tokenizer.decode([idx.item()]) for idx in top_indices]
    top_probs_list = top_probs.tolist()

    return top_tokens, top_probs_list


def plot_top_5_comparison(prompt, heads_to_ablate, model, tokenizer, title="Effect of Ablation"):
    """Generates the dual-bar chart for the report."""
    # Baseline
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out_orig = model(**inputs)
    tokens_orig, probs_orig = get_top_k_tokens(out_orig.logits[0, -1, :], tokenizer)

    # Intervention
    ablator = HeadAblator(model)
    ablator.apply_ablation(heads_to_ablate)
    with torch.no_grad():
        out_int = model(**inputs)
    ablator.remove_hooks()

    tokens_int, probs_int = get_top_k_tokens(out_int.logits[0, -1, :], tokenizer)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.bar(tokens_orig, probs_orig, color='#3498db')
    ax1.set_title(f"Original Model\nPrompt: '{prompt}...'")
    ax1.set_ylabel("Probability")
    ax1.set_ylim(0, 1.0)

    ax2.bar(tokens_int, probs_int, color='#e74c3c')
    ax2.set_title(f"Post-Ablation\nHeads: {heads_to_ablate}")
    ax2.set_ylim(0, 1.0)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    # --- 1. SETUP ---
    # Point this to your local folder if you want to skip the HF Hub warnings
    # MODEL_PATH = r"C:\Users\USER\Desktop\project\GPTExperiment\gpt2"
    MODEL_PATH = "gpt2"  # Change to path if local

    print("Initializing Model and Tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
    model = GPT2LMHeadModel.from_pretrained(MODEL_PATH, output_attentions=True)
    model.eval()

    # --- 2. PART A: THE EXPERIMENT ---
    # This generates the results.csv file
    print("\n--- Starting Full Experiment (Part A) ---")
    run_experiment()

    # --- 3. PART B: DATA ANALYSIS FOR REPORT ---
    print("\n--- Generating Analysis Summary ---")
    res_df = pd.read_csv(OUTPUT_CSV)
    original_data = pd.read_csv("ex1_data.csv")
    res_df['Domain'] = original_data['Domain']

    # Identify the Top 5 most influential layers
    print("\nTop 5 Layers containing 'Knowledge Heads':")
    print(res_df['max_head_layer'].value_counts().head(5))

    # Identify which domain was most impacted by head ablation
    print("\nAverage Probability Drop (Delta) by Domain:")
    print(res_df.groupby('Domain')['condition_a_delta'].mean().sort_values(ascending=False))

    # --- 4. VISUALIZATION ---
    # We find the specific prompt where the single-head ablation (Condition A)
    # had the most dramatic effect.
    print("\n--- Generating Visualization ---")
    best_example_idx = res_df['condition_a_delta'].idxmax()
    row = res_df.loc[best_example_idx]

    target_prompt = original_data.iloc[int(row['prompt_id'])]['Prompt']
    layer = int(row['max_head_layer'])
    head = int(row['max_head_index'])

    print(f"Selecting best example: {target_prompt}")
    print(f"Ablating Head {head} in Layer {layer} (Impact: {row['condition_a_delta']:.2%})")

    plot_top_5_comparison(
        prompt=target_prompt,
        heads_to_ablate={layer: [head]},
        model=model,
        tokenizer=tokenizer,
        title=f"Surgical Ablation: Knowledge Loss in '{original_data.iloc[best_example_idx]['Domain']}'"
    )
