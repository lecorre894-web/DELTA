#!/data/data/com.termux/files/usr/bin/bash
OUT=DELTA_SCELLE.txt
H1=$(sha256sum dqiv.js 2>/dev/null | cut -d' ' -f1)
H2=$(sha256sum dqiv.log 2>/dev/null | cut -d' ' -f1)
{
echo "============= Δ-QI-V — ENREGISTREMENT SCELLÉ ============="
echo "date   : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "auteur : René — AISA, Port-Brillet (Mayenne)"
echo "env    : $(node -v)  /  $(nproc) coeurs"
echo
echo "-- NATURE (vérité) --"
echo "Δ-QI-V V4 : SIMULATION CLASSIQUE Node.js."
echo " - ilots quantiques locaux (vecteur d'etat reel, portes H/CNOT)"
echo " - intrication globale reelle par stabilisateurs (CHP, Gottesman-Knill)"
echo " - echelle en logarithme (ilots / qubits / Hilbert)"
echo " - telemetrie batterie reelle si Termux:API present"
echo "PAS de materiel quantique. PAS d'acceleration quantique."
echo "Aucune mesure en picosecondes, aucun chiffre non calculable."
echo
echo "-- FICHIERS SCELLES (SHA-256) --"
echo "dqiv.js  : ${H1:-absent}  ($(wc -c < dqiv.js 2>/dev/null || echo 0) o)"
echo "dqiv.log : ${H2:-absent}  ($(wc -c < dqiv.log 2>/dev/null || echo 0) o)"
echo
echo "-- VERIFICATION --"
echo "Rejouer : sha256sum dqiv.js"
echo "Le sceau tient si l'empreinte est identique. Un octet change = sceau brise."
echo "========================================================="
} > "$OUT"
S=$(sha256sum "$OUT" | cut -d' ' -f1)
echo "sceau global du present enregistrement : $S" >> "$OUT"
echo "ecrit : $OUT ($(wc -c < "$OUT") o)  sceau dqiv.js : ${H1:-absent}"
