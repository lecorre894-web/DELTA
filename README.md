# DELTA

DELTA est un projet expérimental de calcul classique fondé sur
l'exploitation de la structure des problèmes.

## Principe

Le principe central de DELTA est :

> Le gain vient de la structure, jamais de la magie.

DELTA cherche à éviter les calculs inutiles lorsqu'une structure
mathématique ou physique permet de les éliminer.

Le projet distingue explicitement :

- le travail évité ;
- le travail réellement accéléré ;
- les mesures effectuées ;
- les extrapolations ;
- les limites de principe ;
- les limites dues aux moyens matériels.

## Δ-QI-V V4

La première composante déposée ici est Δ-QI-V V4.

Il s'agit d'une simulation classique Node.js utilisant notamment :

- des îlots quantiques locaux ;
- des portes H/CNOT ;
- des stabilisateurs CHP et Gottesman-Knill ;
- une représentation à échelle logarithmique.

Δ-QI-V V4 n'utilise aucun matériel quantique et ne revendique
aucune accélération quantique.

Le répertoire `delta-qiv-v4/` contient :

- `dqiv.js` — moteur ;
- `dqiv.log` — journal ;
- `sceau.sh` — génération du sceau ;
- `DELTA_SCELLE.txt` — enregistrement d'intégrité SHA-256.

## Intégrité

L'empreinte SHA-256 scellée de `dqiv.js` est :

`934ac42716178bf568d6186a031038bde3e12689a12507dc52384fe9c9d90b0`

Elle a été vérifiée avant la préparation de ce dépôt.

## État du dépôt

Ce dépôt est en cours de reconstruction à partir des différentes
phases documentées de DELTA.

Les composants seront ajoutés progressivement avec leurs benchmarks,
journaux, résultats et limites documentées.

## Auteur

René Le Corre — 2026
