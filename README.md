# DELTA

**DELTA** est un projet expérimental de calcul classique fondé sur l'exploitation de la structure des problèmes.

> **DELTA n'est pas un émulateur quantique. C'est un émulateur classique qui exploite la structure — et qui dit à chaque fois où cette structure s'arrête.**

Le dépôt conserve le code, les benchmarks, les résultats et les journaux expérimentaux afin que les mesures puissent être examinées et reproduites.

---

## État du dépôt

La distribution complète publiée contient **171 fichiers**, comprenant notamment :

- les moteurs et modules DELTA en Python ;
- les programmes de benchmark ;
- les journaux d'exécution (`*_log.txt`) ;
- les résultats structurés (`resultats_*.json`) ;
- les spécifications ;
- les documents de signature et de traçabilité ;
- les expériences spécialisées : structure, échelle, fractal, neurone, cluster, SHA, BTC, qutrit, arène, etc.

Le répertoire principal est :

delta-complet/

Le dépôt conserve également le composant historique :

    delta-qiv-v4/

---

## Principe

DELTA cherche à éviter du travail de calcul lorsqu'une structure exploitable permet une représentation ou un traitement plus compact.

Le gain provient de la structure du problème, et non d'une augmentation artificielle de la puissance du processeur.

Une valeur dite « equivalent-dense » représente du travail dense évité ou son équivalent calculé. Elle ne signifie pas que le processeur physique fonctionne réellement à cette puissance.

Lorsque la structure exploitable disparaît, DELTA peut perdre son avantage. Certains cas peuvent retrouver les coûts classiques du problème sous-jacent, y compris une croissance exponentielle.

---

## Prérequis

- Python 3
- NumPy

Vérification :

    python --version
    python -c "import numpy; print(numpy.__version__)"

---

## Vérification syntaxique

Depuis la racine du dépôt :

    python -m py_compile delta-complet/*.py

Un code retour 0 indique que les fichiers Python concernés ont passé cette vérification syntaxique.

---

## Exécution

Les expériences sont conservées individuellement dans delta-complet/.

Exemple :

    cd delta-complet
    python bench.py

Les autres fichiers bench*.py correspondent aux différentes étapes et expériences du projet.

Avant d'interpréter un résultat, consulter le programme correspondant ainsi que son journal et, lorsqu'il existe, son fichier JSON.

---

## Reproduction des benchmarks

Chaîne générale :

    programme Python
          |
          v
       exécution
          |
          v
    journal *_log.txt
          |
          v
    résultat resultats_*.json

Pour reproduire une expérience :

1. identifier le benchmark concerné ;
2. lire le fichier Python correspondant ;
3. exécuter le benchmark dans un environnement documenté ;
4. conserver la sortie brute ;
5. comparer cette sortie au journal publié ;
6. comparer les valeurs structurées au JSON correspondant.

Les performances dépendent du matériel, du système, de Python, de NumPy, du nombre de cœurs réellement disponibles et des paramètres de l'expérience.

---

## Architecture expérimentale

Le dépôt contient plusieurs générations et extensions de DELTA, notamment :

    delta1 ... delta19
    delta_arene
    delta_atome
    delta_bridge
    delta_btc
    delta_cluster
    delta_diversite
    delta_echelle
    delta_fractal
    delta_neurone
    delta_noeud
    delta_qutrit
    delta_sha
    delta_sigma
    delta_systeme
    delta_un

Chaque expérience doit être interprétée dans son contexte et avec son benchmark associé.

---

## Mesuré et extrapolé

DELTA distingue les résultats effectivement obtenus sur la machine de test des valeurs calculées par extrapolation.

Une valeur extrapolée doit être considérée comme :

    [EXTRAPOLÉ]

et non comme une mesure matérielle directe.

De même, une performance « equivalent-dense » représente une quantité de calcul dense évitée ou équivalente ; elle ne doit pas être confondue avec le débit physique réel du CPU.

---

## Limites

DELTA ne revendique pas d'avantage quantique matériel.

Il s'agit d'un système classique.

Ses gains éventuels dépendent de la structure exploitable des données ou du problème traité.

Dans les situations défavorables, cette structure peut devenir insuffisante et le coût peut rejoindre celui d'une représentation classique dense.

Les limites doivent être distinguées entre :

- limites de principe ;
- limites algorithmiques ;
- limites de mémoire ;
- limites CPU ;
- limites du matériel de test.

Les contre-tests, résultats négatifs et erreurs font partie de l'historique expérimental et ne doivent pas être supprimés pour améliorer artificiellement les résultats.

---

## Statut scientifique

Ce dépôt fournit du code expérimental, des mesures, des résultats et des traces d'exécution.

La présence de ces éléments permet l'inspection et la reproduction des expériences, mais ne constitue pas à elle seule une validation scientifique indépendante ou une validation par les pairs.

Les conclusions doivent rester limitées à ce que démontrent effectivement les expériences reproductibles.

---

## Documentation

Consulter en priorité :

    delta-complet/LISEZMOI.txt
    delta-complet/DELTA_SPECIFICATIONS.txt
    delta-complet/DELTA_BLOC.txt
    delta-complet/DELTA_SIGNATURE.txt

Les fichiers *_log.txt et resultats_*.json fournissent ensuite les traces des différentes expériences.

---

## Philosophie du projet

**Explorer largement, affirmer étroitement.**

DELTA conserve les résultats positifs comme les limites rencontrées afin de séparer ce qui est effectivement mesuré de ce qui est estimé, extrapolé ou encore expérimental.

---

## Installation

DELTA est écrit principalement en Python.

Prérequis :

- Python 3
- NumPy

Installation minimale :

```bash
python -m pip install numpy
```

Cloner le dépôt puis entrer dans la distribution :

```bash
git clone https://github.com/lecorre894-web/DELTA.git
cd DELTA/delta-complet
```
