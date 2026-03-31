"""
metrics.py
----------
Lit results.csv et calcule les deux indicateurs centraux de l'étude :

    SSL  — Score de Sensibilité Linguistique
           Mesure l'écart moyen de performance entre la version anglaise
           et les versions traduites. Un SSL élevé indique que le benchmark
           pénalise les langues non-anglaises indépendamment du contenu.

    R    — Indice de Robustesse par item
           R = 1 − écart-type des scores sur les 3 versions linguistiques
           R = 1.0 → item parfaitement stable (même résultat dans les 3 langues)
           R = 0.0 → item totalement instable

Produit :
    - metrics_summary.json  : résultats agrégés par domaine et globaux
    - items_ranked.csv      : classement des items par indice R croissant

Usage :
    python metrics.py
"""

import csv
import json
import statistics
from pathlib import Path
from collections import defaultdict

RESULTS_FILE  = "results.csv"
SUMMARY_FILE  = "metrics_summary.json"
RANKED_FILE   = "items_ranked.csv"
LANGUAGES     = ["en", "fr", "ar"]
DOMAINS       = ["us_history", "moral_philosophy", "mathematics"]


# ---------------------------------------------------------------------------
# Chargement des résultats
# ---------------------------------------------------------------------------

def load_results(path: str) -> list[dict]:
    results = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Reconvertir les types numériques
            for lang in LANGUAGES:
                row[f"correct_{lang}"] = int(row[f"correct_{lang}"])
            row["robustness_R"] = float(row["robustness_R"])
            results.append(row)
    return results


# ---------------------------------------------------------------------------
# Calcul du SSL — Score de Sensibilité Linguistique
# ---------------------------------------------------------------------------

def compute_ssl(results: list[dict], domain: str | None = None) -> dict:
    """
    SSL global  = score_EN − moyenne(score_FR, score_AR)
    SSL par langue = score_EN − score_LANG

    Un SSL positif signifie que l'anglais obtient de meilleurs résultats,
    ce qui indique un biais en faveur de la langue de construction du benchmark.
    """
    items = results if domain is None else [r for r in results if r["domain"] == domain]
    n = len(items)
    if n == 0:
        return {}

    scores = {}
    for lang in LANGUAGES:
        scores[lang] = sum(r[f"correct_{lang}"] for r in items) / n * 100

    ssl_fr = round(scores["en"] - scores["fr"], 2)
    ssl_ar = round(scores["en"] - scores["ar"], 2)
    ssl_global = round(scores["en"] - (scores["fr"] + scores["ar"]) / 2, 2)

    return {
        "n_items":    n,
        "score_en":   round(scores["en"], 2),
        "score_fr":   round(scores["fr"], 2),
        "score_ar":   round(scores["ar"], 2),
        "ssl_en_fr":  ssl_fr,
        "ssl_en_ar":  ssl_ar,
        "ssl_global": ssl_global,
        "interpretation": interpret_ssl(ssl_global)
    }


def interpret_ssl(ssl: float) -> str:
    if ssl <= 2:
        return "Biais négligeable — item linguistiquement robuste"
    elif ssl <= 10:
        return "Biais modéré — sensibilité linguistique détectable"
    elif ssl <= 20:
        return "Biais fort — la langue affecte significativement le score"
    else:
        return "Biais critique — le benchmark mesure la langue autant que la compétence"


# ---------------------------------------------------------------------------
# Calcul de l'indice R — Robustesse par item
# ---------------------------------------------------------------------------

def compute_robustness_per_item(results: list[dict]) -> list[dict]:
    """
    Pour chaque item, recalcule R et ajoute une catégorie de stabilité.
    Trie par R croissant (les items les plus instables en premier).
    """
    ranked = []
    for r in results:
        scores = [r[f"correct_{lang}"] for lang in LANGUAGES]
        mean   = sum(scores) / len(scores)
        std    = statistics.pstdev(scores)
        R      = round(1 - std, 4)

        if R >= 0.8:
            stability = "stable"
        elif R >= 0.5:
            stability = "instable"
        else:
            stability = "inversé"

        ranked.append({
            "item_id":       r["item_id"],
            "domain":        r["domain"],
            "cultural_load": r["cultural_load"],
            "score_en":      r["correct_en"],
            "score_fr":      r["correct_fr"],
            "score_ar":      r["correct_ar"],
            "robustness_R":  R,
            "stability":     stability
        })

    ranked.sort(key=lambda x: x["robustness_R"])
    return ranked


# ---------------------------------------------------------------------------
# Résumé global
# ---------------------------------------------------------------------------

def compute_summary(results: list[dict]) -> dict:
    summary = {
        "global":  compute_ssl(results),
        "by_domain": {}
    }

    for domain in DOMAINS:
        summary["by_domain"][domain] = compute_ssl(results, domain)

    # Distribution des catégories de stabilité
    stability_counts = defaultdict(int)
    for r in results:
        stability_counts[r["stability"]] += 1

    summary["stability_distribution"] = dict(stability_counts)

    # Top 5 items les plus instables
    ranked = compute_robustness_per_item(results)
    summary["top5_unstable"] = ranked[:5]

    return summary


# ---------------------------------------------------------------------------
# Affichage console
# ---------------------------------------------------------------------------

def print_summary(summary: dict):
    print(f"\n{'='*60}")
    print("  RÉSULTATS — INDICATEURS DE ROBUSTESSE LINGUISTIQUE")
    print(f"{'='*60}\n")

    g = summary["global"]
    print("SCORES GLOBAUX :")
    print(f"  Anglais  (EN) : {g['score_en']}%")
    print(f"  Français (FR) : {g['score_fr']}%")
    print(f"  Arabe    (AR) : {g['score_ar']}%")
    print(f"\nSSL (Score de Sensibilité Linguistique) :")
    print(f"  EN → FR : {g['ssl_en_fr']:+.1f} points")
    print(f"  EN → AR : {g['ssl_en_ar']:+.1f} points")
    print(f"  SSL global : {g['ssl_global']:+.1f} points")
    print(f"  → {g['interpretation']}")

    print(f"\n{'─'*60}")
    print("SCORES PAR DOMAINE :")
    for domain, data in summary["by_domain"].items():
        if not data:
            continue
        print(f"\n  {domain.upper().replace('_', ' ')} :")
        print(f"    EN: {data['score_en']}%  |  FR: {data['score_fr']}%  |  AR: {data['score_ar']}%")
        print(f"    SSL global : {data['ssl_global']:+.1f} → {data['interpretation']}")

    print(f"\n{'─'*60}")
    print("DISTRIBUTION DE STABILITÉ :")
    dist = summary["stability_distribution"]
    total = sum(dist.values())
    for cat, count in sorted(dist.items()):
        pct = round(count / total * 100, 1)
        print(f"  {cat:12s} : {count:2d} items ({pct}%)")

    print(f"\n{'─'*60}")
    print("TOP 5 ITEMS LES PLUS INSTABLES :")
    for item in summary["top5_unstable"]:
        print(f"  [{item['item_id']}] R={item['robustness_R']} "
              f"| EN:{item['score_en']} FR:{item['score_fr']} AR:{item['score_ar']} "
              f"| {item['domain']}")

    print(f"\n{'='*60}")
    print(f"  Fichiers générés : {SUMMARY_FILE} | {RANKED_FILE}")
    print(f"  Étape suivante   : python visualize.py")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not Path(RESULTS_FILE).exists():
        print(f"[ERREUR] Fichier introuvable : {RESULTS_FILE}")
        print("Lance d'abord : python run_experiment.py")
        return

    results  = load_results(RESULTS_FILE)
    summary  = compute_summary(results)
    ranked   = compute_robustness_per_item(results)

    # Sauvegarde JSON
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Sauvegarde CSV classé
    fieldnames = ["item_id", "domain", "cultural_load",
                  "score_en", "score_fr", "score_ar",
                  "robustness_R", "stability"]
    with open(RANKED_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ranked)

    print_summary(summary)


if __name__ == "__main__":
    main()
