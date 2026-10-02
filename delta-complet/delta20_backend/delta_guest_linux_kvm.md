# DELTA — Linux invité sous KVM (Codespace, 2 oct 2026)

Socle : Codespace Xeon Platinum 8573C (2 threads), lui-même machine virtuelle GitHub -> virtualisation IMBRIQUEE.
Invité : noyau Ubuntu 7.0.0-38 + busybox, initramfs, QEMU -enable-kvm -cpu host, 2 vCPU, 1 Go. Démarrage noyau ~1,5 s.

| Test | Socle natif | Invité KVM | Écart |
|---|---|---|---|
| MATMUL FP64 n=384 | 5,97 GF | 2,76 GF | x2,2 |
| MEMCPY | 7,47 Go/s | 1,83 Go/s | x4 |
| FORK | 359 us | 351 us | = |
| Latence repos (moy/pire) | 85 / 1874 us | 198 / 2171 us | - |
| Latence 4 brûleurs (moy/pire) | 117 / 2570 us | 78 / 1715 us | + |

Checksum MATMUL identique socle/invité : 2.391313e+07 (résultat exact).
Référence bac à sable Claude SANS KVM (TCG) : MATMUL x50, MEMCPY x8, FORK x26.

## Sécurité
- Utilisateur 1000 lit /root/secret : REFUSE (prouvé KVM et TCG)
- Remonter en root : REFUSE (prouvé KVM et TCG)
- Bombe de processus limitée à 200 : stoppée à 199 (prouvé TCG)
- OOM (épuisement mémoire) : TCG -> processus tué, noyau vivant ; KVM -> invité FIGÉ, y compris avec ulimit -v. STATUT : OUVERT, cause non identifiée. Hôte non affecté.

## Verdict
Sécurité des privilèges : tenue. Confort sous charge : bon. Coût de la virtualisation imbriquée : x2 à x4 sur calcul et mémoire.
Erreurs de l'auditeur : prédiction x1,0-1,2 sous KVM fausse (imbrication) ; deux hypothèses fausses sur le blocage OOM.
