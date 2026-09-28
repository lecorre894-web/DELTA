'use strict';
// ===== Δ-QI-V V4 — module honnête : îlots locaux + stabilisateur réel + échelle log + télémétrie =====
const fs=require('fs'), os=require('os');
const S=1/Math.SQRT2, H=[[S,0],[S,0],[S,0],[-S,0]];
const pc=x=>{let c=0;while(x){c+=x&1;x>>>=1;}return c;};
const mkR=q=>()=>((q=(q*1103515245+12345)&0x7fffffff)/0x7fffffff);
const fmt=x=>x<1e15?Math.round(x).toLocaleString('fr'):x.toExponential(4);
const e10=L=>{const e=Math.floor(L),m=Math.pow(10,L-e);return m.toFixed(4)+'e+'+fmt(e);};
function jlog(line){fs.appendFileSync('dqiv.log','['+new Date().toISOString()+'] '+line+'\n');}
function cores(){try{if(typeof os.availableParallelism==='function')return os.availableParallelism();return os.cpus().length||1;}catch(e){return 1;}}
function hwtel(){try{const{execSync}=require('child_process');const d=JSON.parse(execSync('termux-battery-status',{encoding:'utf8'}));return '[Temp '+d.temperature+'°C | Bat '+d.percentage+'%]';}catch(e){return '[télémétrie HW non dispo]';}}
// ---- moteur local : îlots (vecteur d'état) ----
function island(n){const N=1<<n;return{n,N,re:new Float64Array(N),im:new Float64Array(N)};}
function gate1(is,t,m){const{re,im,N}=is,b=1<<t;for(let i=0;i<N;i++){if(i&b)continue;const j=i|b;
 const ar=re[i],ai=im[i],br=re[j],bi=im[j];
 re[i]=(m[0][0]*ar-m[0][1]*ai)+(m[1][0]*br-m[1][1]*bi);im[i]=(m[0][0]*ai+m[0][1]*ar)+(m[1][0]*bi+m[1][1]*br);
 re[j]=(m[2][0]*ar-m[2][1]*ai)+(m[3][0]*br-m[3][1]*bi);im[j]=(m[2][0]*ai+m[2][1]*ar)+(m[3][0]*bi+m[3][1]*br);}}
function cnot(is,c,t){const{re,im,N}=is,bc=1<<c,bt=1<<t;for(let i=0;i<N;i++){if((i&bc)&&!(i&bt)){const j=i|bt;
 const r=re[i],m=im[i];re[i]=re[j];im[i]=im[j];re[j]=r;im[j]=m;}}}
function measure(is,rng){const{re,im,N}=is;let r=rng(),a=0,p=0;for(let i=0;i<N;i++){a+=re[i]*re[i]+im[i]*im[i];p=i;if(r<=a)break;}return p;}
function tick(is,rng){is.re.fill(0);is.im.fill(0);is.re[0]=1;gate1(is,0,H);if(is.n>1)cnot(is,0,1);return measure(is,rng);}
function simLocal(Q,isl){const nIsl=Math.ceil(Q/isl),is=island(isl),rng=mkR(Date.now()&0x7fffffff);let sig=0;const t=Date.now();
 for(let i=0;i<nIsl;i++){const p=tick(is,rng);sig+=pc(p)*(0.8+0.4*rng());}return{nIsl,sig,ms:Date.now()-t};}
// ---- moteur stabilisateur : intrication globale réelle (CHP) ----
function chpNew(n){const R=2*n+1,x=new Uint8Array(R*n),z=new Uint8Array(R*n),r=new Uint8Array(R);for(let i=0;i<n;i++){x[i*n+i]=1;z[(n+i)*n+i]=1;}return{n,R,x,z,r};}
function chpCX(s,a,b){const{n,x,z,r,R}=s;for(let i=0;i<R;i++){r[i]^=x[i*n+a]&z[i*n+b]&(x[i*n+b]^z[i*n+a]^1);x[i*n+b]^=x[i*n+a];z[i*n+a]^=z[i*n+b];}}
function chpH(s,a){const{n,x,z,r,R}=s;for(let i=0;i<R;i++){r[i]^=x[i*n+a]&z[i*n+a];const t=x[i*n+a];x[i*n+a]=z[i*n+a];z[i*n+a]=t;}}
function chpG(x1,z1,x2,z2){if(!x1&&!z1)return 0;if(x1&&z1)return z2-x2;if(x1)return z2*(2*x2-1);return x2*(1-2*z2);}
function chpSum(s,h,i){const{n,x,z,r}=s;let u=2*r[h]+2*r[i];for(let j=0;j<n;j++)u+=chpG(x[i*n+j],z[i*n+j],x[h*n+j],z[h*n+j]);
 r[h]=((((u%4)+4)%4)===2)?1:0;for(let j=0;j<n;j++){x[h*n+j]^=x[i*n+j];z[h*n+j]^=z[i*n+j];}}
function chpM(s,a,rng){const{n,x,z,r}=s;let p=-1;for(let i=n;i<2*n;i++)if(x[i*n+a]){p=i;break;}
 if(p>=0){for(let i=0;i<2*n;i++)if(i!==p&&x[i*n+a])chpSum(s,i,p);
  for(let j=0;j<n;j++){x[(p-n)*n+j]=x[p*n+j];z[(p-n)*n+j]=z[p*n+j];}r[p-n]=r[p];
  for(let j=0;j<n;j++){x[p*n+j]=0;z[p*n+j]=0;}z[p*n+a]=1;r[p]=rng()<0.5?0:1;return r[p];}
 const h=2*n;for(let j=0;j<n;j++){x[h*n+j]=0;z[h*n+j]=0;}r[h]=0;for(let i=0;i<n;i++)if(x[i*n+a])chpSum(s,h,i+n);return r[h];}
function scaleInfo(b,d,k,inf){return{Lc:inf?Infinity:d*Math.log10(b),Lq:inf?Infinity:Math.log10(k)+d*Math.log10(b)};}
function ghz(n){const s=chpNew(n),rng=mkR(Date.now()&0x7fffffff),t=Date.now();chpH(s,0);for(let i=1;i<n;i++)chpCX(s,0,i);
 const m0=chpM(s,0,rng);let ok=true,ck=Math.min(n-1,300);for(let i=1;i<=ck;i++)if(chpM(s,i,rng)!==m0)ok=false;
 return{n,m0,ok,ck,mem:((2*n+1)*n*2)/1e6,ms:Date.now()-t};}
function adosse(b,d,k,SI,inf){const{Lc,Lq}=scaleInfo(b,d,k,inf);const n=SI*k,s=chpNew(n),rng=mkR(Date.now()&0x7fffffff),t=Date.now();
 chpH(s,0);for(let i=1;i<SI;i++)chpCX(s,0,i*k);for(let i=0;i<SI;i++)for(let q=1;q<k;q++)chpCX(s,i*k,i*k+q);
 const m0=chpM(s,0,rng);let ok=true,ck=0;for(let i=0;i<SI;i++)for(let q=0;q<k;q++){if(i===0&&q===0)continue;if(chpM(s,i*k+q,rng)!==m0)ok=false;ck++;}
 return{SI,k,n,m0,ok,ck,Lc,Lq,mem:((2*n+1)*n*2)/1e6,ms:Date.now()-t};}
// ---- CLI ----
const A=process.argv,C=A[2]||'help',P=(...a)=>console.log(...a),l2=Math.log10(2);
if(C==='sim'){const hw=hwtel(),Q=+A[3]||1e7,isl=Math.min(Math.max(1,+A[4]||4),20),r=simLocal(Q,isl);
 P('Δ-QI-V sim '+hw+' — '+Q.toLocaleString('fr')+' qubits / '+r.nIsl.toLocaleString('fr')+' îlots');
 P('Δ-signature : '+r.sig.toFixed(2));P('temps       : '+r.ms+' ms');
 jlog('SIM Q='+Q+' isl='+isl+' sig='+r.sig.toFixed(2)+' '+r.ms+'ms '+hw);}
else if(C==='ghz'){const n=+A[3]||2000,r=ghz(n);
 P('Δ-QI-V ghz — '+n+' qubits intriqués (réel CHP)');P('bit épine   : '+r.m0);
 P('vérif '+r.ck+'   : '+(r.ok?'tous = '+r.m0+' ✓':'✗ désync'));P('mémoire     : '+r.mem.toFixed(1)+' Mo, '+r.ms+' ms');
 jlog('GHZ n='+n+' bit='+r.m0+' ok='+r.ok+' '+r.mem.toFixed(1)+'Mo '+r.ms+'ms');}
else if(C==='scale'){const inf=A[3]==='inf',b=+A[3]||100,d=+A[4]||1e9,k=+A[5]||4,{Lc,Lq}=scaleInfo(b,d,k,inf);
 P('Δ-QI-V scale — '+(inf?'∞':b+'^'+d)+', '+k+' q/îlot');
 P('îlots       : '+(inf?'∞':e10(Lc)));P('qubits      : '+(inf?'∞':e10(Lq)));
 if(inf)P('Hilbert     : 2^∞');else if(Lq<300)P('Hilbert 2^q : 10^'+fmt(Math.pow(10,Lq)*l2));
 else P('Hilbert 2^q : 10^(10^'+fmt(Lq+Math.log10(l2))+') — tour');
 jlog('SCALE '+(inf?'inf':b+'^'+d+' k='+k)+' Lq='+(inf?'inf':Lq));}
else if(C==='adosse'){const inf=A[3]==='inf',b=+A[3]||100,d=+A[4]||1e9,k=+A[5]||4,SI=+A[6]||128,r=adosse(b,d,k,SI,inf);
 const pI=inf?'∞':'10^'+fmt(r.Lc),pQ=inf?'∞':'10^'+fmt(r.Lq);
 P('Δ-QI-V adossé — '+SI+' îlots × '+k+' q sur épine GHZ (réel)');P('bit épine   : '+r.m0);
 P('vérif '+r.ck+' q : '+(r.ok?'tous = '+r.m0+' ✓':'✗ désync'));P('mémoire     : '+r.mem.toFixed(1)+' Mo, '+r.ms+' ms');
 P('extension   : '+pI+' îlots / '+pQ+' qubits par symétrie');
 jlog('ADOSSE '+(inf?'inf':b+'^'+d)+' k='+k+' SI='+SI+' bit='+r.m0+' ok='+r.ok);}
else if(C==='hw'){P('Δ-QI-V hw — '+hwtel());P('cœurs       : '+cores());jlog('HW cores='+cores());}
else if(C==='log'){try{P(fs.readFileSync('dqiv.log','utf8'));}catch(e){P('journal vide');}}
else{P('Δ-QI-V V4 — commandes honnêtes :');
 P('  node dqiv.js sim [qubits] [îlot]         îlots locaux + télémétrie');
 P('  node dqiv.js ghz [n]                     intrication globale réelle');
 P('  node dqiv.js scale [b] [d] [k] | inf     îlots / qubits / Hilbert');
 P('  node dqiv.js adosse [b] [d] [k] [SI]|inf îlots adossés à l épine');
 P('  node dqiv.js hw                          cœurs + batterie');
 P('  node dqiv.js log                         journal dqiv.log');}
