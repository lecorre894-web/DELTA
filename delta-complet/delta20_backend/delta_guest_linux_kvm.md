# DELTA — Linux invité sous KVM (Codespace, 2 oct 2026)

Socle : Codespace Xeon Platinum 8573C (2 threads, lui-même VM Azure -> KVM imbriqué).
Invité : noyau Ubuntu 7.0.0-38 + busybox (initramfs), QEMU -enable-kvm -cpu host -mem-prealloc, 2 vCPU, 1 Go.

| Test | Socle natif | Invité KVM | Écart |
|---|---|---|---|
| MATMUL FP64 n=384 | 5,97 GF | 5,17 GF | x1,15 |
| MEMCPY | 7,47 Go/s | 8,80 Go/s | invité plus rapide |
| FORK | 359 us | 94 us | invité x3,8 plus rapide (noyau minimal en RAM) |
| Pages neuves 384 Mo | 2 us/page | 3 us/page | ~natif |
Checksum MATMUL identique socle/invité. Démarrage + tous les tests : 2 s.

## Sécurité (toutes VALIDÉES sous KVM)
- Utilisateur 1000 lit /root/secret : REFUSE
- Remonter en root : REFUSE
- OOM avec ulimit -v : arrêt propre à 384 Mo, noyau vivant
- Isolation : aucun disque hôte, réseau = lo seulement
- Forkbomb (limite 200) : stoppée à 199, système réactif
GUEST_VALIDATION=FIN

## Incident résolu : gel de l'invité
Cause RÉELLE : Codespace saturé (6,7 Go / 7,9 Go utilisés) par des serveurs VS Code fantômes (~310 Mo chacun),
nés des reconnexions du navigateur mobile. Ni KVM, ni l'imbrication, ni le noyau invité.
Remède : redémarrer le Codespace (mémoire dispo 1,2 -> 5,7 Go) + -mem-prealloc + vérifier `free -m` avant toute VM.
Erreurs de l'auditeur : trois hypothèses successives partiellement fausses (enlisement OOM, pages en imbrication) ;
la mesure (`free`, `ps`) a tranché.

## Contre-mesure Claude (bac à sable SANS KVM, émulation TCG)
MATMUL x37, MEMCPY x9, FORK x19 plus lents que natif ; sécurité identique, validée.
Conclusion : KVM fait passer le coût d'un Linux invité de x9-x37 à ~x1.
