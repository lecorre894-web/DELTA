# REGISTRE DELTA — point d'entrée commun (Claude, ChatGPT, Mistral, Qwen)
Etat : 1cb56cf | backend delta-complet/delta20_backend | Codespace curly orbit (1 coeur HT, L2 1,3 Mio)
Valide : marque jusqu'a x92 | tuiles telephone x7,86 | moteur L1 x7,9 (87 GB/s logiques, 1 coeur) | DDR5 V2 dual 86,7 GB/s | chaine run_all 16/16
Regles : mesurer avant d'annoncer ; separer MESURE / EMULE / MODELISE ; tuile 8-16 Kio en L1 (source+destination comptees) ; restrict pour vectoriser
DELTA 1 = reference ; DELTA 2 = cluster gros grain, donnees locales (emulation x1,30 a 2 noeuds)
Securite : SECRETS=0 avant commit ; pas de port ouvert ; pas d'automatisme ; cles IBM uniquement en secrets Codespaces
Methode : une conversation par session ; journal pousse ici puis lu ; fin de session = mise a jour de ce fichier
