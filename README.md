# Stress-Test de Robustesse Linguistique — MMLU
## Exercice 2 — Master IREN, Université Paris Dauphine PSL

---

## Question de recherche
Le benchmark MMLU produit-il des scores différents selon la langue de formulation,
indépendamment de la capacité réelle du modèle testé ?

---

## Architecture du projet

```
stress_test/
├── corpus.json          ← 30 questions MMLU (EN + FR + AR)
├── run_experiment.py    ← Soumet les questions à Mistral via Ollama
├── metrics.py           ← Calcule SSL et indice R
├── visualize.py         ← Génère les 3 graphiques
└── README.md            ← Ce fichier

Fichiers générés après exécution :
├── results.csv          ← Réponses brutes du modèle
├── metrics_summary.json ← Indicateurs agrégés
├── items_ranked.csv     ← Items classés par robustesse
├── barplot_scores.png   ← Scores par langue et domaine
├── heatmap_robustness.png ← Heatmap des réponses
└── histogram_R.png      ← Distribution de l'indice R
```

---

## Installation (une seule fois)

### 1. Installer Ollama
- Télécharger sur https://ollama.com
- Télécharger le modèle : `ollama pull mistral`

### 2. Installer les librairies Python
```bash
pip install requests matplotlib pandas seaborn numpy
```

---

## Exécution (dans l'ordre)

```bash
# Étape 1 : lancer Ollama (dans un terminal séparé)
ollama serve

# Étape 2 : lancer l'expérience
python run_experiment.py

# Étape 3 : calculer les indicateurs
python metrics.py

# Étape 4 : générer les graphiques
python visualize.py
```

---

## Indicateurs

### SSL — Score de Sensibilité Linguistique
```
SSL = score_EN − moyenne(score_FR, score_AR)
```
- SSL ≤ 2  : biais négligeable
- SSL ≤ 10 : biais modéré
- SSL ≤ 20 : biais fort
- SSL > 20 : biais critique

### R — Indice de Robustesse par item
```
R = 1 − écart-type(scores EN, FR, AR)
```
- R ≥ 0.8 : item stable
- 0.5 ≤ R < 0.8 : item instable
- R < 0.5 : item inversé (résultats contradictoires selon la langue)

---

## Conditions de contrôle
- Modèle fixe : Mistral 7B (ollama)
- Température : 0.0 (déterministe)
- Seed : 42
- Prompt système identique pour les 3 langues
- Source des questions : MMLU (Hendrycks et al., 2020)

---

## Domaines testés
| Domaine            | Charge culturelle | N questions |
|--------------------|-------------------|-------------|
| Histoire américaine | Élevée           | 10          |
| Philosophie morale  | Élevée           | 10          |
| Mathématiques       | Faible (contrôle)| 10          |
