import matplotlib
matplotlib.use("TkAgg")

import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import pandas as pd
import torch
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from ex1_main import HeadAblator, get_subject_token_index


MODEL_NAME = "gpt2"
DATA_PATH = "ex1_data.csv"
RESULTS_PATH = "ID_results.csv"


def heads_for_condition(prompt: str, subject: str, model, tokenizer, condition: str):
    """
    Returns heads to ablate as {layer_idx: [head_idx, ...]} for condition A/B/C.
    Condition definitions follow the assignment:
      - A: single strongest head focusing on subject
      - B: top 3 strongest heads focusing on subject
      - C: all heads with attention weight > 0.25 focusing on subject
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    subj_idx = get_subject_token_index(prompt, subject, tokenizer)

    # attentions: tuple(layers) each: (batch, heads, seq, seq)
    all_attentions = torch.stack(outputs.attentions)  # (layers, batch, heads, seq, seq)
    weights = all_attentions[:, 0, :, -1, subj_idx]  # (layers, heads)
    flat = weights.flatten()
    top = torch.argsort(flat, descending=True)

    if condition == "A":
        best = top[0].item()
        layer, head = divmod(best, 12)
        return {layer: [head]}

    if condition == "B":
        top3 = top[:3].tolist()
        d = {}
        for f in top3:
            l, h = divmod(int(f), 12)
            d.setdefault(l, []).append(h)
        return d

    if condition == "C":
        idxs = torch.where(flat > 0.25)[0].tolist()
        d = {}
        for f in idxs:
            l, h = divmod(int(f), 12)
            d.setdefault(l, []).append(h)
        return d

    raise ValueError(f"Unknown condition: {condition}")


def make_top5_figure(prompt: str, heads_to_ablate, model, tokenizer, title: str):
    """
    Creates a matplotlib Figure (no plt.show) with a double-bar chart comparing:
      - baseline top-5 next-token probs
      - post-ablation probs for the same 5 tokens
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out_orig = model(**inputs)

    logits_orig = out_orig.logits[0, -1, :]
    probs_orig = torch.softmax(logits_orig, dim=-1)
    top_ids = torch.topk(probs_orig, 5).indices.tolist()

    ablator = HeadAblator(model)
    ablator.apply_ablation(heads_to_ablate)
    with torch.no_grad():
        out_int = model(**inputs)
    ablator.remove_hooks()

    probs_int = torch.softmax(out_int.logits[0, -1, :], dim=-1)

    tokens = [tokenizer.decode([tid]) for tid in top_ids]
    before = [probs_orig[tid].item() for tid in top_ids]
    after = [probs_int[tid].item() for tid in top_ids]

    fig = Figure(figsize=(7.6, 4.4), dpi=100)
    ax = fig.add_subplot(111)

    x = np.arange(len(tokens))
    width = 0.38
    ax.bar(x - width / 2, before, width, label="Original", color="#3498db")
    ax.bar(x + width / 2, after, width, label="Post-Ablation", color="#e74c3c")

    ax.set_xticks(x)
    ax.set_xticklabels(tokens, rotation=20, ha="right")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1.0)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


class ResultsViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("GPT-2 Attention Ablation Viewer")
        self.geometry("980x640")

        # Load data
        try:
            self.data_df = pd.read_csv(DATA_PATH)
        except Exception as e:
            raise SystemExit(f"Failed to read {DATA_PATH}: {e}")

        try:
            self.res_df = pd.read_csv(RESULTS_PATH)
        except Exception as e:
            raise SystemExit(f"Failed to read {RESULTS_PATH}: {e}")

        self.merged = self.res_df.merge(
            self.data_df.reset_index().rename(columns={"index": "prompt_id"}),
            on="prompt_id",
            how="left",
        )

        # Load model once
        self.status_var = tk.StringVar(value="Loading GPT-2 (first time may download)...")
        self.update_idletasks()
        self.tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
        self.model = GPT2LMHeadModel.from_pretrained(MODEL_NAME, output_attentions=True)
        self.model.eval()
        self.status_var.set("Ready.")

        self._build_ui()
        self._populate_list()

    def _build_ui(self):
        outer = ttk.Frame(self, padding=10)
        outer.pack(fill="both", expand=True)

        left = ttk.Frame(outer)
        left.pack(side="left", fill="y")

        right = ttk.Frame(outer)
        right.pack(side="right", fill="both", expand=True)

        ttk.Label(left, text="Prompts").pack(anchor="w")
        self.listbox = tk.Listbox(left, width=48, height=28, exportselection=False)
        self.listbox.pack(fill="y", expand=True)
        self.listbox.bind("<<ListboxSelect>>", self._on_select)

        controls = ttk.Frame(left, padding=(0, 10, 0, 0))
        controls.pack(fill="x")

        ttk.Label(controls, text="Condition").grid(row=0, column=0, sticky="w")
        self.condition = tk.StringVar(value="A")
        self.condition_combo = ttk.Combobox(
            controls,
            width=6,
            textvariable=self.condition,
            values=["A", "B", "C"],
            state="readonly",
        )
        self.condition_combo.grid(row=0, column=1, sticky="w", padx=(8, 0))

        self.plot_btn = ttk.Button(controls, text="Plot top-5 before/after", command=self._plot_selected)
        self.plot_btn.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(1, weight=0)

        self.status = ttk.Label(left, textvariable=self.status_var, foreground="#444")
        self.status.pack(anchor="w", pady=(10, 0))

        ttk.Label(right, text="Details").pack(anchor="w")
        self.details = tk.Text(right, height=10, wrap="word")
        self.details.pack(fill="x")
        self.details.configure(state="disabled")

        ttk.Label(right, text="Visualization").pack(anchor="w", pady=(10, 0))
        self.figure_container = ttk.Frame(right)
        self.figure_container.pack(fill="both", expand=True)
        self._canvas = None

    def _populate_list(self):
        self.listbox.delete(0, tk.END)
        for _, row in self.merged.iterrows():
            pid = int(row["prompt_id"])
            prompt = str(row["Prompt"])
            domain = str(row.get("Domain", ""))
            label = f"{pid:02d} [{domain}] {prompt}"
            self.listbox.insert(tk.END, label)

        if self.listbox.size() > 0:
            self.listbox.selection_set(0)
            self.listbox.event_generate("<<ListboxSelect>>")

    def _selected_row(self):
        sel = self.listbox.curselection()
        if not sel:
            return None
        idx = int(sel[0])
        return self.merged.iloc[idx]

    def _on_select(self, _evt=None):
        row = self._selected_row()
        if row is None:
            return

        lines = []
        lines.append(f"prompt_id: {int(row['prompt_id'])}")
        lines.append(f"Domain: {row.get('Domain', '')}")
        lines.append("")
        lines.append(f"Prompt: {row.get('Prompt', '')}")
        lines.append(f"Subject: {row.get('Subject Word(s)', '')}")
        lines.append(f"Target Token: {row.get('Target Token', '')}")
        lines.append("")
        lines.append(f"baseline_prob: {row.get('baseline_prob', '')}")
        lines.append(f"condition_a_delta: {row.get('condition_a_delta', '')}")
        lines.append(f"condition_b_delta: {row.get('condition_b_delta', '')}")
        lines.append(f"condition_c_delta: {row.get('condition_c_delta', '')}")
        lines.append("")
        lines.append(f"max_head_layer (A): {row.get('max_head_layer', '')}")
        lines.append(f"max_head_index (A): {row.get('max_head_index', '')}")

        self.details.configure(state="normal")
        self.details.delete("1.0", tk.END)
        self.details.insert("1.0", "\n".join(lines))
        self.details.configure(state="disabled")

    def _plot_selected(self):
        row = self._selected_row()
        if row is None:
            messagebox.showwarning("No selection", "Please select a prompt.")
            return

        prompt = str(row["Prompt"])
        subject = str(row["Subject Word(s)"])
        cond = self.condition.get().strip().upper()

        try:
            self.status_var.set(f"Computing heads for condition {cond}...")
            self.update_idletasks()
            heads = heads_for_condition(prompt, subject, self.model, self.tokenizer, cond)

            title = f"Condition {cond} — Top-5 next-token probabilities (before vs after)"
            fig = make_top5_figure(prompt, heads, self.model, self.tokenizer, title=title)

            # Replace existing canvas
            for child in self.figure_container.winfo_children():
                child.destroy()
            self._canvas = FigureCanvasTkAgg(fig, master=self.figure_container)
            self._canvas.draw()
            self._canvas.get_tk_widget().pack(fill="both", expand=True)
            self.status_var.set(f"Plotted condition {cond}. Heads: {heads}")
        except Exception as e:
            self.status_var.set("Ready.")
            messagebox.showerror("Plot failed", str(e))


if __name__ == "__main__":
    app = ResultsViewer()
    app.mainloop()

