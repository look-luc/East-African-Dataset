import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd


def escape_latex(text: str) -> str:
  """Escapes special LaTeX characters while preserving unicode punctuation."""
  if not isinstance(text, str):
    return text
  # Replace unprotected special chars, avoiding double-escaping
  # Escaping &, %, $, #, _, {, }
  replacements = {
      "&": r"\&",
      "%": r"\%",
      "$": r"\$",
      "#": r"\#",
      "_": r"\_",
  }
  # Simple regex pass for unescaped characters
  for char, escaped in replacements.items():
    text = text.replace(char, escaped)
  return text


def parse_latex_eval_table(tex_path: str) -> pd.DataFrame:
  """Parses a multirow LaTeX evaluation table into a clean DataFrame."""
  text = Path(tex_path).read_text(encoding="utf-8")

  # Match each multirow proverb block
  block_pattern = re.compile(
      r"\\multirow\[t\]\{4\}\{(?P<proverb>.*?)\}\s*&\s*Levenshtein Distance\s*&\s*(?P<ld_mod>\[.*?\])\s*&\s*(?P<ld_man>\[.*?\])\s*\\\\"
      r".*?Levenshtein Normalized Sim\s*&\s*(?P<lns_mod>\[.*?\])\s*&\s*(?P<lns_man>\[.*?\])\s*\\\\"
      r".*?ChrF\s*&\s*(?P<chrf_mod>\[.*?\])\s*&\s*(?P<chrf_man>\[.*?\])\s*\\\\"
      r".*?Sentence Bert Embeddings\s*&\s*(?P<sbert_mod>[-0-9.]+)\s*&\s*(?P<sbert_man>[-0-9.]+)\s*\\\\",
      re.DOTALL,
  )

  rows = []
  for match in block_pattern.finditer(text):
    d = match.groupdict()

    # Safely evaluate and convert list representations to scalar means
    ld_mod = np.mean(ast.literal_eval(d["ld_mod"]))
    ld_man = np.mean(ast.literal_eval(d["ld_man"]))
    lns_mod = np.mean(ast.literal_eval(d["lns_mod"]))
    lns_man = np.mean(ast.literal_eval(d["lns_man"]))
    chrf_mod = ast.literal_eval(d["chrf_mod"])[0]
    chrf_man = ast.literal_eval(d["chrf_man"])[0]
    sbert_mod = float(d["sbert_mod"])
    sbert_man = float(d["sbert_man"])

    rows.append({
        "Proverb": escape_latex(d["proverb"].strip()),
        ("Model vs. Corrected", "Lev. Dist"): round(ld_mod, 2),
        ("Model vs. Corrected", "Lev. Sim"): round(lns_mod, 3),
        ("Model vs. Corrected", "ChrF"): round(chrf_mod, 3),
        ("Model vs. Corrected", "SBERT"): round(sbert_mod, 3),
        ("Manual vs. Corrected", "Lev. Dist"): round(ld_man, 2),
        ("Manual vs. Corrected", "Lev. Sim"): round(lns_man, 3),
        ("Manual vs. Corrected", "ChrF"): round(chrf_man, 3),
        ("Manual vs. Corrected", "SBERT"): round(sbert_man, 3),
    })

  df = pd.DataFrame(rows).set_index("Proverb")
  df.columns = pd.MultiIndex.from_tuples(df.columns)
  return df


# Map your language files here (e.g., Ekegusii, Luganda, Samia)
language_files = {
    "Ekegusii": "./ekegusii_table.tex",
    "Luganda": "./luganda_table.tex",
    "Gikuyu": "./gikuyu_table.tex",
}

for lang, filepath in language_files.items():
  if not Path(filepath).exists():
    print(f"Skipping {lang}: file not found at {filepath}")
    continue

  df = parse_latex_eval_table(filepath)

  # Compute corpus-level mean summary row
  mean_row = df.mean().round(3)
  df.loc["Mean"] = mean_row

  # Export cleanly using pandas to_latex with multi-column headers
  latex_output = df.to_latex(
      caption=f"Translation Evaluation Metrics for {lang}",
      label=f"tab:{lang.lower()}_eval",
      float_format="%.3f",
      multicolumn_format="c",
      escape=False,  # Safe because we manually sanitized strings via escape_latex
  )

  out_path = f"clean_{lang.lower()}_table.tex"
  Path(out_path).write_text(latex_output, encoding="utf-8")
  print(f"Successfully generated {out_path} with {len(df) - 1} proverbs.")
