/* DELTA Control : tout est genere depuis l'API. Aucune liste de moteurs codee en dur. */
const $=id=>document.getElementById(id);
const PROV={PHYSICAL_MEASURED:["mesure","Mesuré physiquement"],PHYSICAL_MEASURED_ON_HOST:["hote","Mesuré sur l'hôte"],
  LOGICAL_EMULATED:["emule","Émulé (logique)"],EMULATED_ON_XEON:["emule","Émulé sur Xeon"],VIRTUALIZED:["virtuel","Virtualisé"],
  PHYSICAL_QPU:["qpu","QPU physique"],UNKNOWN:["inconnu","Provenance inconnue"]};
const prov=c=>PROV[c]||(/EMUL/.test(c)?["emule",c]:/VIRT/.test(c)?["virtuel",c]:/QPU/.test(c)?["qpu",c]:/PHYSICAL/.test(c)?["mesure",c]:["inconnu",c]);
const couleur=k=>getComputedStyle(document.documentElement).getPropertyValue("--"+k).trim();
const esc=s=>String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const nb=(v,d=3)=>v==null?null:v!==0&&Math.abs(v)<1?v.toLocaleString("fr-FR",{maximumSignificantDigits:3}):Math.abs(v)>=1e5?v.toLocaleString("fr-FR",{maximumFractionDigits:0}):v.toLocaleString("fr-FR",{maximumFractionDigits:d});
const quand=t=>t?new Date(t*1000).toLocaleTimeString("fr-FR"):"N/A";
const api=async(u,o)=>{const r=await fetch(u,o);let d=null;try{d=await r.json()}catch(e){}return{ok:r.ok,code:r.status,d}};
let filtre="Tous",comps=[],dernierLog=0,enCours=false;
function toast(t){const e=$("toast");e.textContent=t;e.classList.add("vu");clearTimeout(e._t);e._t=setTimeout(()=>e.classList.remove("vu"),3200)}

const LIGNES=[["temps","Temps",""],["gflops","Calcul flottant","GFLOPS",1],["debit","Débit",null,1],
  ["qubits","Qubits","",0],["shots","Tirs","",0],["fidelite","Fidélité min","",3],["jobs","Jobs","",0],["threads","Threads","",0],
  ["cores","Cœurs","",0],["iterations_s","Itérations","/s",1],["ram","RAM","Go",1],["vram","VRAM","Go",1],["sm","SM","",0],["warps","Warps","",0],
  ["backend","Backend",""],["job_id","Job IBM",""],["checksum","Checksum",""],["signature","Signature",""],["sig","Signature",""]];
function carte(c){
  const [k,lib]=prov(c.classification),m=c.metrics||{};
  const rows=LIGNES.filter(([cle])=>cle==="temps"||m[cle]!=null).map(([cle,label,unite,d])=>{
    if(cle==="temps")return m.us==null?`<div class="m"><span>Temps</span><b class="na">N/A</b></div>`
      :`<div class="m"><span>Temps</span><b>${nb(m.us,1)} µs<small> · ${nb(m.ms,3)} ms</small></b></div>`;
    let v=m[cle];
    if(typeof v==="number")v=nb(v,d);const u=cle==="debit"?" "+esc(m.unite||""):unite?" "+unite:"";
    return `<div class="m"><span>${label}</span><b>${esc(v)}${esc(u)}</b></div>`}).join("");
  const au=m.autres?Object.entries(m.autres).map(([a,b])=>`<div class="m autre"><span>${esc(a)}</span><b>${esc(nb(b,4))}</b></div>`).join(""):"";
  const st=c.status&&c.status!=="N/A"?` · ${esc(c.status)}`:"";
  return `<article class="carte"><div class="bande" style="background:${couleur(k)}" title="${esc(lib)}">${esc(c.classification)}</div>
  <div class="corps"><div class="type">${esc(c.type)}${st}</div><h3>${esc(c.name)}</h3>${rows}${au}
  <div class="pied">${esc(c.source)}${c.classification_source&&c.classification_source!=="json"?" · classé par "+esc(c.classification_source):""}</div></div></article>`}

function dessinerComposants(){
  const types=["Tous",...[...new Set(comps.map(c=>c.type))].sort()];
  if(!types.includes(filtre))filtre="Tous";
  $("filtres").innerHTML=types.map(t=>`<button role="tab" aria-selected="${t===filtre}" data-t="${esc(t)}">${esc(t)} <small>${t==="Tous"?comps.length:comps.filter(c=>c.type===t).length}</small></button>`).join("");
  const vis=filtre==="Tous"?comps:comps.filter(c=>c.type===filtre);
  $("cartes").innerHTML=vis.length?vis.map(carte).join(""):`<p class="vide">Aucun résultat JSON trouvé. Lance une commande ci-dessus : ses mesures apparaîtront ici.</p>`;
  $("note-comp").textContent=comps.length+" composant(s), lus depuis les JSON du dépôt";
  const n={};comps.forEach(c=>{const k=prov(c.classification)[0];n[k]=(n[k]||0)+1});
  $("spectre").innerHTML=Object.entries(n).map(([k,v])=>`<span style="flex-grow:${v};background:${couleur(k)}" title="${v}"></span>`).join("");
  const vus=[...new Set(comps.map(c=>prov(c.classification)[0]))];
  $("legende").innerHTML=Object.values(PROV).filter(([k],i,a)=>a.findIndex(x=>x[0]===k)===i&&vus.includes(k))
    .map(([k,lib])=>`<li><i style="background:${couleur(k)}"></i>${lib} <b>${n[k]||0}</b></li>`).join("");
}
$("filtres").addEventListener("click",e=>{const b=e.target.closest("button");if(b){filtre=b.dataset.t;dessinerComposants()}});

async function lancer(c){
  if(c.dangerous&&!confirm(`« ${c.label} » consomme une ressource réelle (${c.category}). Lancer quand même ?`))return;
  const r=await api(`/api/run/${encodeURIComponent(c.id)}${c.dangerous?"?confirm=1":""}`,{method:"POST"});
  toast(r.ok?`${c.label} : lancé`:`${c.label} : ${r.d?.error||"refusé"}`);rafraichir()}
function dessinerCommandes(cmds){
  const dispo=cmds.filter(c=>c.available);
  $("cmds").innerHTML=dispo.length?dispo.map((c,i)=>`<button class="cmd${c.dangerous?" danger":""}${c.running?" encours":""}" data-i="${i}" ${c.running?"disabled":""}>
    <b>${esc(c.label)}</b><code>${esc(c.argv.join(" "))}</code><span class="etat">${c.running?"En cours…":esc(c.category)}</span></button>`).join("")
    :`<p class="vide">Aucune commande disponible : ajoute-les dans <code>delta_ui/command_registry.json</code>.</p>`;
  $("cmds").onclick=e=>{const b=e.target.closest(".cmd");if(b)lancer(dispo[+b.dataset.i])};
  const absentes=cmds.length-dispo.length;$("t-cmd").title=absentes?absentes+" commande(s) enregistrée(s) mais programme absent":"";}

function dessinerStatut(s){
  $("pouls").classList.add("on");$("etat").textContent=`En ligne · mis à jour à ${quand(s.timestamp)}`;
  const lv=s.last_validation?`${quand(s.last_validation.mtime)}<small> ${esc(s.last_validation.file)}</small>`:"N/A";
  const dj=s.last_job?`${esc(s.last_job.status)}<small> ${esc(s.last_job.label)} · ${nb(s.last_job.duration_ms,1)} ms</small>`:"N/A";
  $("chiffres").innerHTML=[["Composants",s.components],["Modules",`${s.modules}<small> + ${s.unknown} autres</small>`],
    ["Jobs",`${s.jobs.total}<small> · ${s.jobs.running} en cours · ${s.jobs.failed} échec(s)</small>`],
    ["DSPC",s.dspc?s.dspc.detected+"<small> détecté(s)</small>":"N/A"],["Dernier job",dj],["Dernière validation",lv]]
    .map(([a,b])=>`<div><dt>${a}</dt><dd>${b}</dd></div>`).join("");
  enCours=s.jobs.running>0;$("arret").hidden=!enCours}

function dessinerJobs(js){
  $("jobs").innerHTML=js.length?js.slice(0,12).map(j=>`<div class="job"><span class="pastille ${esc(j.status)}">${esc(j.status)}</span>
   <span>${esc(j.label||j.command_id)}</span><span class="duree">${j.duration_ms==null?"…":nb(j.duration_ms,1)+" ms"}</span>
   <small>${quand(j.started)} · ${j.duration_us==null?"en cours":nb(j.duration_us,0)+" µs"} · code ${j.returncode??"N/A"} · ${esc(j.job_id)}</small></div>`).join("")
   :`<p class="vide">Aucun job pour l'instant.</p>`}

async function lireLog(){
  const r=await api("/api/log?since="+dernierLog);if(!r.ok)return;const L=r.d.lines;if(!L.length)return;
  const pre=$("log");if(dernierLog===0)pre.textContent="";dernierLog=r.d.last;
  pre.insertAdjacentHTML("beforeend",L.map(l=>{const t=esc(l.line);const cl=/^\[ui\]/.test(l.line)?"ui":/(FAIL|ECHEC|ERREUR|Error|Traceback|FAUX)/.test(l.line)?"ko":/(=OK|PASSED|IDENTIQUE|\bexact\b|\[OK\])/.test(l.line)?"ok":"";
    return cl?`<span class="${cl}">${t}</span>\n`:t+"\n"}).join(""));
  if($("suivre").checked)pre.scrollTop=pre.scrollHeight}

async function rafraichir(){
  try{
    const [s,c,k,j,m]=await Promise.all([api("/api/status"),api("/api/components"),api("/api/commands"),api("/api/jobs"),api("/api/modules")]);
    dessinerStatut(s.d);comps=c.d;dessinerComposants();dessinerCommandes(k.d);dessinerJobs(j.d);
    const g={};m.d.modules.forEach(x=>(g[x.type]=g[x.type]||[]).push(x.name));if(m.d.unknown.length)g["Autres composants (UNKNOWN)"]=m.d.unknown.map(x=>x.name);
    $("note-mod").textContent=m.d.modules.length+m.d.unknown.length+" fichiers Python";
    $("mods").innerHTML=Object.entries(g).map(([t,l])=>`<div class="groupe"><h4>${esc(t)} (${l.length})</h4><p>${l.map(esc).join(", ")}</p></div>`).join("");
  }catch(e){$("pouls").classList.remove("on");$("etat").textContent="Serveur injoignable : relance python3 delta_ui/server.py"}}
$("arret").onclick=async()=>{const r=await api("/api/stop",{method:"POST"});toast(r.d.stopped+" job(s) arrêté(s)");rafraichir()};
rafraichir();
setInterval(rafraichir,3000);(function boucle(){lireLog().finally(()=>setTimeout(boucle,enCours?800:3000))})();
