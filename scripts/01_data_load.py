"""
Étape 1 — Jointure textes OCR + métadonnées CSV
+ export des stats complètes dans outputs/data_info.txt
"""

import os
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm


# =====================
# Setup projet
# =====================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

os.chdir(PROJECT_ROOT)
sys.path.append(str(PROJECT_ROOT))


# =====================
# Paramètres
# =====================

YEARS = ["1973", "1978", "1981", "1988", "1993"]

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

CSV_PATH = DATA_DIR / "archelect_search.csv"
TEXT_DIR = DATA_DIR
OUTPUT_PATH = DATA_DIR / "corpus_joined.csv"
STATS_PATH = OUTPUT_DIR / "data_info.txt"


# =====================
# Chargement CSV
# =====================

def load_metadata(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]

    df = df[df["contexte-election"] == "législatives"].copy()
    df["year"] = df["date"].str[:4]
    df = df[df["year"].isin(YEARS)].copy()

    return df


# =====================
# Mapping fichiers texte
# =====================

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


# =====================
# Lecture texte
# =====================

def load_text_file(path):
    if pd.isna(path):
        return None
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None


# =====================
# Main
# =====================

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

    # =====================
    # Sauvegarde CSV
    # =====================

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    # =====================
    # Stats
    # =====================

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

        # =====================
        # PAR ANNEE — soutien
        # =====================

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

        # =====================
        # PAR ANNEE — liste
        # =====================

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

    print("\n Corpus sauvegardé :", OUTPUT_PATH)
    print(" Stats sauvegardées :", STATS_PATH)
    print("\n--- QUALITÉ DE JOINTURE ---")
    


if __name__ == "__main__":
    main()