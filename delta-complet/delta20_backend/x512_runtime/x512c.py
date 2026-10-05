"""Traducteur X512 v0 -> C AVX-512 (compile d'avance). Usage : python3 x512c.py prog.x512 > prog.c"""
import sys,re
def tr(src):
    out=[];depth=1;cnt=[];ind=lambda:"  "*depth
    for ln in src.splitlines():
        ln=ln.split("#")[0].strip()
        if not ln:continue
        op,*a=re.split(r"[\s,]+",ln)
        X=lambda r:"X[%s]"%r[1:];R=lambda r:"R[%s]"%r[1:]
        if op=="loop":out.append(ind()+"for(long %s=0;%s<%s;%s++){"%(a[0],a[0],a[1],a[0]));depth+=1;cnt.append(None)
        elif op=="end":depth-=1;cnt.pop();out.append(ind()+"}")
        elif op=="xld":out.append(ind()+"%s=_mm512_load_si512(%s+8*(%s));NI++;"%(X(a[0]),a[1],a[2]))
        elif op=="xst":out.append(ind()+"_mm512_store_si512(%s+8*(%s),%s);NI++;"%(a[0],a[1],X(a[2])))
        elif op=="xtern":out.append(ind()+"%s=_mm512_ternarylogic_epi64(%s,%s,%s,%s);NI++;"%(X(a[0]),X(a[1]),X(a[2]),X(a[3]),a[4]))
        elif op.startswith("xadd."):w={"q":"epi64","d":"epi32","w":"epi16","b":"epi8"}[op[-1]];out.append(ind()+"%s=_mm512_add_%s(%s,%s);NI++;"%(X(a[0]),w,X(a[1]),X(a[2])))
        elif op=="xpop":out.append(ind()+"P[%s]=_mm512_add_epi64(P[%s],_mm512_popcnt_epi64(%s));NI++;"%(a[0][1:],a[0][1:],X(a[1])))
        elif op=="tpop":out.append(ind()+"P[%s]=_mm512_add_epi64(P[%s],_mm512_popcnt_epi64(_mm512_ternarylogic_epi64(%s,%s,%s,%s)));NI++;"%(a[0][1:],a[0][1:],X(a[1]),X(a[2]),X(a[3]),a[4]))
        elif op=="macc":n=a[0][1:];out.append(ind()+"macc(&M[%s],%s,%s);NI++;"%(n,X(a[1]),X(a[2])))
        elif op=="mread":out.append(ind()+"R[%s]+=mread(&M[%s]);NI++;"%(a[0][1:],a[1][1:]))
        elif op=="flush":out.append(ind()+"R[%s]+=_mm512_reduce_add_epi64(P[%s]);P[%s]=_mm512_setzero_si512();NI++;"%(a[0][1:],a[0][1:],a[0][1:]))
        else:raise SystemExit("instruction inconnue : "+op)
    return "\n".join(out)
HDR=r'''#include <immintrin.h>
#include <stdint.h>
/* micro-registres v2 = Harley-Seal (modele Rene v1) : plans ones/twos/fours/eights + retenues en attente ; 1 popcount par 16 vecteurs */
typedef struct{__m512i m1,m2,m4,m8,ta,fa,ea;uint64_t S;int n;}Micro;
static inline uint64_t pc(__m512i v){return _mm512_reduce_add_epi64(_mm512_popcnt_epi64(v));}
#define CSA_(h,l,a,b,c) do{__m512i _a=(a),_b=(b),_c=(c);h=_mm512_ternarylogic_epi64(_a,_b,_c,0xE8);l=_mm512_ternarylogic_epi64(_a,_b,_c,0x96);}while(0)
static inline void mvide(Micro*m){m->S+=8*pc(m->m8)+4*pc(m->m4)+2*pc(m->m2)+pc(m->m1);
  if(m->n&1)m->S+=2*pc(m->ta); if(m->n&2)m->S+=4*pc(m->fa); if(m->n&4)m->S+=8*pc(m->ea);
  m->m1=m->m2=m->m4=m->m8=_mm512_setzero_si512();m->n=0;}
static inline void macc(Micro*m,__m512i a,__m512i b){__m512i t,f,e,s;
  CSA_(t,m->m1,m->m1,a,b);
  if(!(m->n&1)){m->ta=t;m->n++;return;}
  CSA_(f,m->m2,m->m2,m->ta,t);
  if(!(m->n&2)){m->fa=f;m->n++;return;}
  CSA_(e,m->m4,m->m4,m->fa,f);
  if(!(m->n&4)){m->ea=e;m->n++;return;}
  CSA_(s,m->m8,m->m8,m->ea,e);
  m->S+=16*pc(s);m->n=0;}            /* 16 vecteurs absorbes -> 1 seul popcount */
static inline uint64_t mread(Micro*m){mvide(m);uint64_t s=m->S;m->S=0;return s;}
'''
if __name__=="__main__":
    src=open(sys.argv[1]).read();fn=sys.argv[2] if len(sys.argv)>2 else "prog"
    print(HDR+"static void %s(__m512i*X,uint64_t*R,__m512i*P,Micro*M,const uint64_t*A,const uint64_t*B,const uint64_t*C,long*NIp,long NV){long NI=0;\n%s\n  *NIp+=NI;}\n"%(fn,tr(src)))
