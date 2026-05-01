"""
Step 11 — Topic sanity digest.

Per topic: top c-TF-IDF words, three representative chunks, and three
heuristic flags (GENERIC_BOILERPLATE, REPETITIVE_WORDS, DUPLICATE_CHUNKS).
Used for the manual labelling notebook and as a coherence sanity check.
"""

import ast
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = PROJECT_ROOT / "outputs"

TOPIC_INFO_PATH = OUTPUTS / "07_bertopic" / "topic_info.csv"

STEP_DIR = OUTPUTS / "11_visualizations" / "sanity"
REPORTS_DIR = STEP_DIR / "reports"
STEP_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def parse_repr(value):
    if not isinstance(value, str):
        return []
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []


def truncate(text, n=220):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    return text[:n] + ("..." if len(text) > n else "")


def flag_topic(top_words, chunks):
    flags = []
    if not top_words:
        return ["NO_WORDS"]

    generic = {"circonscription", "candidat", "élection", "election", "majorité",
               "majorite", "politique", "pays", "français", "francais", "vote",
               "voter", "monsieur", "madame", "union"}
    generic_count = sum(1 for w in top_words[:8] if w.split()[0] in generic)
    if generic_count >= 4:
        flags.append("GENERIC_BOILERPLATE")

    word_tokens = [w.split()[0] for w in top_words[:8]]
    if len(set(word_tokens)) <= 3:
        flags.append("REPETITIVE_WORDS")

    if chunks:
        prefixes = [c[:60].lower() for c in chunks if isinstance(c, str)]
        if len(set(prefixes)) == 1 and len(prefixes) > 1:
            flags.append("DUPLICATE_CHUNKS")
    return flags


def main():
    print(f"Loading {TOPIC_INFO_PATH}...")
    info = pd.read_csv(TOPIC_INFO_PATH)
    info["words"] = info["Representation"].apply(parse_repr)
    info["docs"] = info["Representative_Docs"].apply(parse_repr)
    info_sorted = info.sort_values("Count", ascending=False)

    flag_rows = []
    digest_path = REPORTS_DIR / "topic_digest.txt"
    with open(digest_path, "w", encoding="utf-8") as f:
        f.write("=== TOPIC SANITY DIGEST ===\n")
        f.write(f"Source: {TOPIC_INFO_PATH}\n")
        f.write(f"Topics: {len(info)}\n\n")

        for _, row in info_sorted.iterrows():
            tid = row["Topic"]
            count = row["Count"]
            name = row.get("Name", "")
            words = row["words"][:8]
            docs = row["docs"][:3]
            flags = flag_topic(words, docs)
            flag_str = ", ".join(flags) if flags else "OK"
            flag_rows.append({"topic": tid, "count": count, "name": name, "flags": flag_str})

            f.write(f"\n{'=' * 78}\n")
            f.write(f"Topic {tid:>4}  |  count={count:>6}  |  flags: {flag_str}\n")
            f.write(f"{'=' * 78}\n")
            f.write(f"Name:  {name}\n")
            f.write(f"Words: {' | '.join(str(w) for w in words)}\n\n")
            for i, doc in enumerate(docs, 1):
                f.write(f"  [Rep doc {i}] {truncate(doc)}\n\n")

    flags_df = pd.DataFrame(flag_rows)
    flags_df.to_csv(STEP_DIR / "topic_flags.csv", index=False, encoding="utf-8-sig")

    n_ok = (flags_df["flags"] == "OK").sum()
    print(f"Done → {STEP_DIR}  |  OK: {n_ok}  flagged: {len(flags_df) - n_ok}")


if __name__ == "__main__":
    main()
