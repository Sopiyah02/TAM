import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files
from io import StringIO

CATEGORIES = {
    'PU':   'Perceived Usefulness',
    'PEOU': 'Perceived Ease of Use',
    'ATU':  'Attitude Toward Using',
    'BI':   'Behavioral Intention to Use',
    'AU':   'Actual Use',
}


def interpret(mean):
    """4-point Likert interpretation (1=SD, 2=D, 3=A, 4=SA)."""
    if mean >= 3.26: return "Strongly Agree"
    if mean >= 2.51: return "Agree"
    if mean >= 1.76: return "Disagree"
    return "Strongly Disagree"


def visualize_tam(summary, filename="tam_category_means.png"):
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(summary['Code'], summary['Mean'], color='steelblue',
                  edgecolor='black', alpha=0.85)
    ax.axhline(2.5, color='orange', linestyle='--', linewidth=1.5,
               label='Midpoint (2.5)')
    ax.set_ylim(1, 4)
    ax.set_ylabel('Mean (1–4)')
    ax.set_title('TAM Category Means')
    ax.legend(loc='lower right')
    for b, m in zip(bars, summary['Mean']):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.05, f"{m:.2f}",
                ha='center', fontsize=10)
    fig.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    files.download(filename)


def compute_tam(df, remove_uniform=True):
    """
    Compute TAM category scores (mean of items per category) on a 4-point scale.
    Columns must be named PU1.., PEOU1.., ATU1.., BI1.., AU1.. (case-insensitive).
    """
    df = df.copy()
    df.columns = df.columns.str.strip().str.upper()

    # Map category -> its item columns
    groups = {}
    for code in CATEGORIES:
        cols = [c for c in df.columns if re.fullmatch(rf"{code}\d+", c)]
        if not cols:
            raise ValueError(f"No columns found for {code}. Expected {code}1, {code}2, ...")
        groups[code] = cols

    item_cols = [c for cols in groups.values() for c in cols]
    df = df[item_cols].apply(pd.to_numeric, errors='coerce')

    if not df.stack().between(1, 4).all():
        raise ValueError("All answers must be between 1 and 4.")

    # Optional: remove straight-lining (same answer for every item)
    uniform_mask = df.nunique(axis=1) == 1
    removed = df[uniform_mask]
    if remove_uniform:
        df = df[~uniform_mask].copy()

    # Per-respondent category scores
    scores = pd.DataFrame({code: df[cols].mean(axis=1) for code, cols in groups.items()})
    scores['TAM_Overall'] = scores.mean(axis=1)

    # Summary
    rows = []
    for col in scores.columns:
        m, s, n = scores[col].mean(), scores[col].std(), scores[col].count()
        moe = 1.96 * s / np.sqrt(n) if n > 1 else 0
        rows.append({
            'Code': col,
            'Category': CATEGORIES.get(col, 'Overall TAM'),
            'Items': len(groups.get(col, [])) or '-',
            'Mean': round(m, 2),
            'SD': round(s, 2),
            'MoE (95%)': round(moe, 2),
            'Interpretation': interpret(m),
        })
    summary = pd.DataFrame(rows)

    print("===== TECHNOLOGY ACCEPTANCE MODEL (TAM) ANALYSIS =====")
    print(f"Total responses loaded : {len(df) + (len(removed) if remove_uniform else 0)}")
    if remove_uniform:
        print(f"Uniform responses removed : {len(removed)}")
    print(f"Respondents analyzed   : {len(df)}\n")
    print(summary.to_string(index=False))

    visualize_tam(summary[summary['Code'] != 'TAM_Overall'])

    scores.round(2).to_csv("tam_scores.csv", index=False)
    summary.to_csv("tam_summary.csv", index=False)
    files.download("tam_scores.csv")
    files.download("tam_summary.csv")

    return scores, summary

import csv

def load_tam_blocks(text):
    """Parse stacked TAM blocks (title row 'Name (CODE)', header row, data rows)
    into one wide DataFrame with columns PU1..PU4, PEOU1.., etc."""
    blocks, current = {}, None
    for row in csv.reader(StringIO(text)):
        row = [c.strip() for c in row]
        if not any(row):
            current = None
            continue
        m = re.search(r"\((PU|PEOU|ATU|BI|AU)\)", row[0], re.I)
        if m and not any(row[1:]):
            current = m.group(1).upper()
            blocks[current] = []
            continue
        if row[0].lower().startswith('no'):   # header row
            continue
        if current:
            blocks[current].append(row)

    frames = []
    for code, rows in blocks.items():
        f = pd.DataFrame(rows).set_index(0)
        f.columns = [f"{code}{i}" for i in range(1, f.shape[1] + 1)]
        frames.append(f)
    return pd.concat(frames, axis=1).dropna().apply(pd.to_numeric)
# -----------------------------
# Upload CSV in Google Colab
# -----------------------------
uploaded = files.upload()
csv_file = list(uploaded.keys())[0]
df = load_tam_blocks(uploaded[csv_file].decode('utf-8-sig'))

scores, summary = compute_tam(df)