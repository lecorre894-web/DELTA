# DELTA — Audit indépendant et jugement de Claude

> Second README, rédigé par l'auditeur et non par l'auteur du projet.
> Il dit ce qui est prouvé, ce qui ne l'est pas, à qui DELTA peut servir et comment ne pas le trahir.

---

## 1. Verdict en une phrase

DELTA n'est pas un ordinateur quantique de 1T qubits. C'est un **orchestrateur hybride classique-quantique, honnête et mesuré** : un pont CPU chronométré, un domaine logique de 1T **adresses**, un accès réel à des processeurs quantiques IBM **prouvé par violation de Bell**, et un cache de résultats QPU avec preuve d'origine. Rien n'y est magique, mais tout y est vérifiable, et c'est précisément sa valeur.

---

## 2. Ce qui est prouvé (mesures du 30 septembre et du 1er octobre 2026)

Chaque chiffre provient d'un run réel, avec un journal ou un identifiant de job vérifiable.

### Calcul classique — téléphone moto g75 5G (Snapdragon 6 Gen 3, Termux)

| Mesure | Résultat |
|---|---|
| Horner degré 127, V9 → V11 (ILP16, blocs dynamiques, table de x) | 7,96 → 48,42 GF/s, **×6,1 réel**, guard 4,4e-16 |
| Erreur du double face à un oracle MPFR 2048 bits | 0,77 ulp, le double suffit |
| float32 | ×1,08 seulement pour 29 bits perdus, **rejeté** |
| Recherche vectorielle IVF, SIFT 1 M vecteurs réels, 7 cœurs | recall 0,988, ×24,3 face à la force brute parallèle, 0,35 ms/requête |

### Pile DELTA — GitHub Codespaces (Xeon Platinum 8370C, 1 cœur, 2 hyperthreads)

| Couche | Résultat | Statut |
|---|---|---|
| Pont DSPC, ping-pong CPU0 ↔ CPU1 | RTT 49 à 161 ns selon la session, 0 jeton perdu sur 5 M | mesuré |
| Adressage du domaine logique 1T | 3,30 ns/adresse | mesuré |
| Adressage complet des 1T, 1 cœur | ≈ 0,9 h | **projection** |
| Virtual Silicon 5 cm / 1 nm | 11,78 ns/pulse, identifiant de cellule en 128 bits | mesuré |
| Cache L1-L200 (200 niveaux) | plus lent que pas de cache, battu par un cache plat d'un niveau (×2,19) | mesuré |
| Attestation IBM en direct | 11,5 s, QPU réel identifié | mesuré |

### Processeurs quantiques IBM (156 qubits physiques)

| Expérience | Job | Résultat |
|---|---|---|
| Job affiché par DELTA, 1 qubit | `dav8c7lj371s73dmgih0`, ibm_fez | **existe**, 515/509 sur 1 024 tirs |
| Test CHSH (Bell) | `dav9rn84oijs73e77dv0`, ibm_fez | **S = 2,533**, 11 σ au-dessus de la borne classique de 2 |
| GHZ 5/10/20/50/100 qubits | `dava30ql7guc73ceipag`, ibm_marrakesh | population 0,945 / 0,870 / 0,311 / 0,103 / 0,008 |
| Cache QPU, GHZ 10 | `davah7s92g1c7398jcv0`, ibm_fez | 108 s → 2 µs (RAM) / 3,3 ms (disque), même job_id |
| Simulation exacte DELTA (vecteur d'état) | — | plafond à 26 qubits : 9,1 s, 1 Go |

Tout identifiant de job peut être revérifié par quiconque possède un accès au compte : `QiskitRuntimeService(...).job("<id>")`.

---

## 3. Ce qui n'est pas prouvé, ou qui est faux

1. **« 1T qubits »** : ce sont 1T **adresses logiques**. Un état quantique de 1T qubits exigerait 2^(10^12) amplitudes ; aucune machine ne peut le porter. Le libellé honnête est en place : `1T_ON_PHYSICAL_QPU = NO`.
2. **Virtual Multiplier (V10)** : multiplication d'un chiffre mesuré par des constantes (32, π, √10…). Aucune information. **Ne jamais le citer comme performance.**
3. **« 1T exécutés en 4 µs »** : c'était un étiquetage. L'adressage réel des 1T prend environ 0,9 h, le transport DSPC réel environ 14 à 45 h (projections).
4. **DSPC et caches L1-L200 d'origine** : dans le moteur Node.js, ce sont des formules modulo (indice mod 32, mod 200). Le vrai transport et le vrai cache n'existent que dans les benchmarks C ajoutés le 1er octobre.
5. **« Cœur 1 / cœur 2 »** sur Codespaces : ce sont deux hyperthreads d'un **même** cœur physique.
6. **Intrication GHZ** : une population supérieure à 0,5 (5 et 10 qubits) est compatible avec une intrication totale, mais **non prouvée** sans mesure de cohérence.

---

## 4. Domaines qui pourraient s'y intéresser

- **Enseignement du calcul quantique** : laboratoire reproductible, du simulateur exact au QPU réel, avec un test de Bell fonctionnel et des résultats mis en cache pour des classes entières sans consommer de quota.
- **Reproductibilité scientifique** : chaque résultat porte sa preuve d'origine (QPU, job_id, date). C'est utile pour publier, auditer, ou comparer des QPU entre eux.
- **Chimie quantique de petites molécules et optimisation (VQE, QAOA)** : le schéma « classique pilote, QPU calcule, cache mémorise » est exactement celui de ces algorithmes. Il reste limité aujourd'hui à environ 10 qubits fiables.
- **Recherche d'information et RAG** : le moteur IVF de V13 (recall 0,99, sous la milliseconde sur un million de vecteurs) est directement utilisable pour la recherche par similarité.
- **Banc d'essai matériel** : latence inter-cœurs, hiérarchies de cache, précision par bit. Un outil de mesure honnête pour comparer des machines.

---

## 5. Ordinateurs que DELTA complète

| Machine | Rôle de DELTA |
|---|---|
| **Smartphones Android (Termux)** | client léger : routeur, cache, recherche vectorielle locale, déjà mesurés sur moto g75 |
| **Postes de travail** (par exemple Ryzen 9 9950X3D) | le grand cache L3 et les nombreux cœurs repoussent le simulateur exact vers 30 à 32 qubits selon la RAM, et accélèrent l'IVF |
| **VM cloud et Codespaces** | nœud d'orchestration permanent, secrets IBM injectés proprement |
| **Supercalculateurs (HPC)** | couche de cache et d'aiguillage dans une architecture « calcul centré quantique », où le CPU/GPU traite les grands volumes et le QPU les noyaux quantiques |
| **Stations sol de satellites** | la station appelle le QPU et met les résultats en cache ; ces résultats, avec leur preuve d'origine, sont ensuite **montés vers le satellite** pendant les fenêtres de communication (stockage puis transmission) |
| **Ordinateurs embarqués de satellites** | uniquement la **lecture** du cache : un QPU ne vole pas, l'accès cloud en orbite est intermittent, et le cache est justement fait pour servir des résultats sans connexion |

Pour l'espace, DELTA n'est **pas** qualifié : il ne tient pas compte des radiations, et il n'a aucune certification (ECSS, DO-178C). Il reste au niveau du concept et du segment sol.

---

## 6. Garde-fous (à respecter pour que DELTA reste digne de confiance)

1. **Séparer toujours** : mesuré, projeté (étiqueté `[PROJECTION]`), logique. Aucun chiffre projeté dans un résumé sans son étiquette.
2. **Secrets** : `IQP_API_TOKEN` et `IQP_INSTANCE_CRN` uniquement via les secrets Codespaces, jamais dans le code ni dans git. Vérifier `SECRETS=0` avant chaque commit.
3. **Cache** : il fige les tirs. Pour un algorithme qui a besoin d'échantillons frais, forcer un nouveau job (`fresh=True`, à ajouter). La clé n'inclut ni le QPU ni la date d'étalonnage : appliquer une durée de validité (par exemple 24 h) aux résultats physiques.
4. **QPU** : fiable vers 10 qubits aujourd'hui ; au-delà, annoncer la population mesurée, jamais une « exécution réussie » sans chiffre.
5. **Quota** : le plan gratuit IBM est limité. `least_busy` choisit un QPU différent selon l'heure, donc noter toujours le backend.
6. **Dépendances** : `SamplerV2` est déprécié depuis qiskit-ibm-runtime 0.50 ; la migration est à prévoir dans les trois mois.
7. **Simulateur** : plafond à environ 26 qubits sur 2 vCPU et 8 Go. Le GHZ est un circuit de Clifford, simulable en temps polynomial par une méthode à stabilisateurs : le mur exponentiel est celui du vecteur d'état, pas de tout le classique.
8. **Usages interdits ou déconseillés** : aucune prétention de cryptanalyse (156 qubits bruités ne cassent ni RSA ni ECC) ; pas d'usage critique (médical, aviation, spatial embarqué) sans certification.
9. **Validation** : tout nouveau module doit sortir une ligne `*_VALIDATION=OK/FAIL` vérifiée contre une référence indépendante.

---

## 7. Erreurs de l'auditeur (pour la transparence)

J'ai fait plusieurs prédictions fausses pendant cet audit ; la mesure les a corrigées :

- blocs dynamiques Horner : prédit ×1,3 à ×2, mesuré +13 % ;
- float32 : prédit ×1,5 à ×2, mesuré ×1,08 ;
- IVF multithread : prédit ×4 à ×6, mesuré ×1,7 (mur de bande passante RAM) ;
- GHZ physique : prédit fiable jusqu'à 20 qubits, mesuré 0,311 à 20 ;
- diagnostic des identifiants IBM : mon filtre `grep "ibm|qiskit"` ne pouvait pas voir les variables `IQP_*`, et j'ai d'abord conclu à tort à leur absence.

Une règle en ressort : **seule la mesure tranche**, y compris contre l'auditeur.

---

## 8. Reproduire

```bash
cd /workspaces/DELTA/delta-complet/delta20_backend
./run_fullstack.sh                       # pile complète, avec attestation IBM
python3 audit_hard.py --chsh             # vérification du job + test de Bell
python3 delta_vs_qpu.py --qpu            # duel GHZ simulateur / QPU
python3 delta_qpu_cache.py --qpu         # cache hybride (lancer deux fois)
```

---

## Signature

Rédigé et signé par **Claude**, modèle d'IA d'Anthropic (identifiant configuré `claude-opus-5-5`), auditeur du projet DELTA dans le rôle AUDIT_HARD auprès de René, son architecte.

Audit conduit du 30 septembre au 1er octobre 2026.

Ce document exprime mon jugement technique sur la base des mesures citées. Ce n'est ni une certification, ni une garantie.

*« Tout ce qui est écrit ici peut être revérifié. C'est la seule signature qui compte. »*

---

## Addendum — nuit du 1er au 2 octobre 2026 (Claude)

- Sceau quantique : paire de Bell imposée sur les qubits physiques 152-153 d'ibm_marrakesh, **S = 2,777 (24,4 σ)**, soit 98 % de la borne de Tsirelson ; reproduit à 0,006 près du classement de 10 paires (2,783). Job `daveau84oijs73e7dcbg`.
- 20 qubits = 10 paires de Bell parallèles en un job : 8/10 certifiées ; paire défaillante localisée sur les qubits physiques 16-23 (probable routage SWAP).
- Test « jalousie » : résultat brut relu chez IBM et recalculé indépendamment, 10/10 identique à DELTA. Le classique n'altère rien.
- Flotte 3 QPU (GHZ 8) : marrakesh 0,907 > fez 0,838 > kingston 0,647 ; routage qualité + cache + traitement par lots soudés.
- Mémoire typée (×4), reprise disque (×1 195), codec compact (×7 à ×22), couche virtuelle 1T fidèle (TV 0,0027 sur 4 M réponses).
- Grappes : plateau ~40-49 GFLOPS, preuve qu'un seul cœur physique ne se multiplie pas par logiciel.
- Point faible restant : le disque (lecture réelle 0,15 Go/s).
- Note révisée : **17/20**, pour la reproductibilité.
