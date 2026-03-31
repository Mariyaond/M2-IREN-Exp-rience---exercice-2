"""
run_experiment.py
-----------------
Soumet chaque question du corpus à Mistral via Ollama dans les 3 langues,
enregistre les réponses et produit un fichier results.csv.

Prérequis :
    - Ollama installé (https://ollama.com)
    - Modèle téléchargé : ollama pull mistral
    - Python : pip install requests

Usage :
    python run_experiment.py
"""

import json
import csv
import time
import requests
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_URL  = "http://localhost:11434/api/generate"
MODEL_NAME = "phi3"
TEMPERATURE = 0.0        # Déterministe : même question = même réponse
CORPUS_FILE = "corpus.json"
RESULTS_FILE = "results.csv"

SYSTEM_PROMPT = """You are answering a multiple choice question.
Read the question carefully and respond with ONLY the letter of the correct answer: A, B, C, or D.
Do not explain. Do not add any text. Just the letter."""

LANGUAGES = ["en", "fr", "ar"]

# ---------------------------------------------------------------------------
# Fonctions utilitaires
# ---------------------------------------------------------------------------

def load_corpus(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_prompt(question: str, choices: list[str]) -> str:
    """Formate la question et les choix en un seul prompt."""
    choices_text = "\n".join(choices)
    return f"{question}\n\n{choices_text}"


def query_ollama(prompt: str) -> str:
    """
    Envoie une requête à Ollama et retourne la réponse du modèle.
    Retourne 'ERROR' en cas d'échec de connexion.
    """
    payload = {
        "model": MODEL_NAME,
        "prompt": f"{SYSTEM_PROMPT}\n\n{prompt}",
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "seed": 42
        }
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()
        raw = response.json().get("response", "").strip()
        # Extraire uniquement la première lettre A/B/C/D
        for char in raw.upper():
            if char in ["A", "B", "C", "D"]:
                return char
        return "INVALID"
    except requests.exceptions.ConnectionError:
        print("\n[ERREUR] Ollama n'est pas lancé.")
        print("Démarre Ollama avec : ollama serve")
        return "ERROR"
    except Exception as e:
        print(f"\n[ERREUR] {e}")
        return "ERROR"


def extract_letter(text: str) -> str:
    """Extrait la lettre de réponse depuis le texte du modèle."""
    for char in text.upper():
        if char in ["A", "B", "C", "D"]:
            return char
    return "INVALID"


# ---------------------------------------------------------------------------
# Expérience principale
# ---------------------------------------------------------------------------

def run_experiment():
    corpus = load_corpus(CORPUS_FILE)
    questions = corpus["questions"]
    total = len(questions) * len(LANGUAGES)
    done = 0

    results = []

    print(f"\n{'='*60}")
    print(f"  STRESS-TEST ROBUSTESSE LINGUISTIQUE — MMLU")
    print(f"  Modèle : {MODEL_NAME} | Température : {TEMPERATURE}")
    print(f"  Questions : {len(questions)} × {len(LANGUAGES)} langues = {total} requêtes")
    print(f"{'='*60}\n")

    for q in questions:
        item_id = q["id"]
        domain  = q["domain"]
        cultural_load = q["cultural_load"]

        print(f"[{item_id}] {domain} (charge culturelle : {cultural_load})")

        row = {
            "item_id":       item_id,
            "domain":        domain,
            "cultural_load": cultural_load,
        }

        for lang in LANGUAGES:
            version   = q[lang]
            prompt    = format_prompt(version["question"], version["choices"])
            expected  = version["answer"]

            response  = query_ollama(prompt)
            is_correct = int(response == expected)

            row[f"response_{lang}"] = response
            row[f"correct_{lang}"]  = is_correct

            status = "✓" if is_correct else "✗"
            print(f"  [{lang.upper()}] Attendu: {expected} | Obtenu: {response} {status}")

            done += 1
            time.sleep(0.3)   # Pause légère pour ne pas saturer Ollama

        # Calcul de l'indice R pour cet item
        scores = [row[f"correct_{lang}"] for lang in LANGUAGES]
        mean   = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        std    = variance ** 0.5
        row["robustness_R"] = round(1 - std, 4)

        # Catégorie de stabilité
        if row["robustness_R"] >= 0.8:
            row["stability"] = "stable"
        elif row["robustness_R"] >= 0.5:
            row["stability"] = "instable"
        else:
            row["stability"] = "inverse"

        results.append(row)
        print(f"  → Indice R = {row['robustness_R']} ({row['stability']})\n")

    # ---------------------------------------------------------------------------
    # Sauvegarde CSV
    # ---------------------------------------------------------------------------

    fieldnames = [
        "item_id", "domain", "cultural_load",
        "response_en", "correct_en",
        "response_fr", "correct_fr",
        "response_ar", "correct_ar",
        "robustness_R", "stability"
    ]

    with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{'='*60}")
    print(f"  Expérience terminée. Résultats sauvegardés dans : {RESULTS_FILE}")
    print(f"{'='*60}\n")

    # Résumé rapide
    print("RÉSUMÉ PAR LANGUE :")
    for lang in LANGUAGES:
        correct = sum(r[f"correct_{lang}"] for r in results)
        pct = round(correct / len(results) * 100, 1)
        print(f"  {lang.upper()} : {correct}/{len(results)} ({pct}%)")

    print("\nRÉSUMÉ PAR DOMAINE :")
    for domain in ["us_history", "moral_philosophy", "mathematics"]:
        domain_items = [r for r in results if r["domain"] == domain]
        for lang in LANGUAGES:
            correct = sum(r[f"correct_{lang}"] for r in domain_items)
            pct = round(correct / len(domain_items) * 100, 1)
            print(f"  {domain} [{lang.upper()}] : {correct}/{len(domain_items)} ({pct}%)")

    print("\nÉtape suivante : python metrics.py")


if __name__ == "__main__":
    run_experiment()
