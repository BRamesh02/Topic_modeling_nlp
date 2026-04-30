"""
Étape 1 — Jointure des fichiers OCR avec les métadonnées CSV. Filtre sur les
législatives 1973–1993 et exporte un descriptif dans data_info.txt.
"""

import os
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

os.chdir(PROJECT_ROOT)
sys.path.append(str(PROJECT_ROOT))


YEARS = ["1973", "1978", "1981", "1988", "1993"]

DATA_DIR = PROJECT_ROOT / "data"
STEP_DIR = PROJECT_ROOT / "outputs" / "01_data_load"
REPORTS_DIR = STEP_DIR / "reports"
FIG_DIR = STEP_DIR / "figures"

STEP_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "archelect_search.csv"
TEXT_DIR = DATA_DIR
OUTPUT_PATH = STEP_DIR / "corpus_joined.csv"
STATS_PATH = REPORTS_DIR / "data_info.txt"


def load_metadata(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]

    df = df[df["contexte-election"] == "législatives"].copy()
    df["year"] = df["date"].str[:4]
    df = df[df["year"].isin(YEARS)].copy()

    return df


def build_id_to_path(text_root: Path) -> dict:
    id_to_path = {}

    for year in YEARS:
        year_dir = text_root / year
        if not year_dir.exists():
            print(f"Dossier absent : {year_dir}")
            continue

        for p in tqdm(list(year_dir.rglob("*.txt")), desc=f"Indexation {year}"):
            id_to_path[p.stem] = p

    return id_to_path


def load_text_file(path):
    if pd.isna(path):
        return None
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None


def main():
    print("Chargement des métadonnées...")
    df = load_metadata(CSV_PATH)

    print("Mapping fichiers texte...")
    id_to_path = build_id_to_path(TEXT_DIR)

    df["text_path"] = df["id"].map(id_to_path)
    df["text"] = df["text_path"].apply(load_text_file)

    df["text_length"] = df["text"].apply(
        lambda x: len(x.split()) if isinstance(x, str) else 0
    )

    # Colonnes utiles
    cols = [
        "id", "date", "year", "contexte-tour",
        "titulaire-liste", "titulaire-soutien",
        "titulaire-sexe", "titulaire-prenom", "titulaire-nom",
        "text", "text_length"
    ]
    cols = [c for c in cols if c in df.columns]
    df = df[cols].copy()


    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")


    with open(STATS_PATH, "w", encoding="utf-8") as f:

        f.write("===== DATASET INFO =====\n\n")
        f.write(f"Documents total : {len(df)}\n")
        f.write(f"Textes trouvés : {df['text'].notna().sum()}\n")
        f.write(f"Textes manquants : {df['text'].isna().sum()}\n")
        f.write(f"Taux manquant : {df['text'].isna().mean():.3f}\n\n")

        f.write("===== LONGUEUR DES TEXTES =====\n\n")
        f.write(str(df["text_length"].describe()))
        f.write("\n\n")

        f.write("===== PAR ANNEE =====\n\n")
        f.write(str(df["year"].value_counts().sort_index()))
        f.write("\n\n")

        if "contexte-tour" in df.columns:
            f.write("===== PAR TOUR =====\n\n")
            f.write(str(df["contexte-tour"].value_counts()))
            f.write("\n\n")

        if "titulaire-soutien" in df.columns:
            f.write("===== TOP PARTIS (soutien) =====\n\n")
            f.write(str(df["titulaire-soutien"].value_counts().head(20)))
            f.write("\n\n")

        if "titulaire-liste" in df.columns:
            f.write("===== TOP LISTES =====\n\n")
            f.write(str(df["titulaire-liste"].value_counts().head(20)))
            f.write("\n\n")

        # PAR ANNEE — soutien

        if "titulaire-soutien" in df.columns:
            f.write("===== TITULAIRE-SOUTIEN PAR ANNEE =====\n\n")
            soutien_year = pd.crosstab(df["year"], df["titulaire-soutien"], dropna=False)
            f.write(soutien_year.to_string())
            f.write("\n\n")

            f.write("===== TOP 10 TITULAIRE-SOUTIEN PAR ANNEE =====\n\n")
            for year, sub in df.groupby("year"):
                f.write(f"\n--- {year} ---\n")
                f.write(sub["titulaire-soutien"].value_counts(dropna=False).head(10).to_string())
                f.write("\n")

        # PAR ANNEE — liste

        if "titulaire-liste" in df.columns:
            f.write("\n===== TITULAIRE-LISTE PAR ANNEE =====\n\n")
            liste_year = pd.crosstab(df["year"], df["titulaire-liste"], dropna=False)
            f.write(liste_year.to_string())
            f.write("\n\n")

            f.write("===== TOP 10 TITULAIRE-LISTE PAR ANNEE =====\n\n")
            for year, sub in df.groupby("year"):
                f.write(f"\n--- {year} ---\n")
                f.write(sub["titulaire-liste"].value_counts(dropna=False).head(10).to_string())
                f.write("\n")


    import matplotlib.pyplot as plt

    # 1. Documents per year
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df["year"].value_counts().sort_index()
    ax.bar(counts.index.astype(str), counts.values, color="#4a7ab5")
    ax.set_title("Documents per election year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of documents")
    for i, v in enumerate(counts.values):
        ax.text(i, v + max(counts.values) * 0.01, str(v), ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "documents_per_year.png", dpi=150)
    plt.close()

    # 2. Top 15 parties (titulaire-soutien)
    if "titulaire-soutien" in df.columns:
        top_parties = df["titulaire-soutien"].value_counts().head(15)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(top_parties))[::-1], top_parties.values, color="#4a7ab5")
        ax.set_yticks(range(len(top_parties))[::-1])
        ax.set_yticklabels([str(s)[:50] for s in top_parties.index], fontsize=9)
        ax.set_title("Top 15 partis (titulaire-soutien)")
        ax.set_xlabel("Number of documents")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "top_parties.png", dpi=150)
        plt.close()

    # 3. Text length distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    valid = df[df["text_length"] > 0]["text_length"]
    ax.hist(valid, bins=60, color="#4a7ab5", alpha=0.85)
    ax.axvline(valid.median(), color="red", linestyle="--", label=f"median = {int(valid.median())}")
    ax.set_xlabel("Text length (words)")
    ax.set_ylabel("Number of documents")
    ax.set_title("Document length distribution")
    ax.set_xlim(0, valid.quantile(0.99))
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "text_length_distribution.png", dpi=150)
    plt.close()

    print("\n Corpus sauvegardé :", OUTPUT_PATH)
    print(" Stats sauvegardées :", STATS_PATH)
    print(" Figures sauvegardées :", FIG_DIR)


if __name__ == "__main__":
    main()