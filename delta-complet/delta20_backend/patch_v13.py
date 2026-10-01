f="delta_v12_1_SECTION_V10_VIRTUAL_MULTIPLIER_EXP.c"; s=open(f).read()
a="int main(void) {"
b="    int all_ok=1;"
fn=r'''/* ============================================================
   §DELTA V13 — IVF VECTOR SEARCH ON REAL DATA (SIFT)
   Measured only: GT float MT, brute uint8 MT, k-means IVF MT.
   Own validation; never alters the V12.1 canonical validation.
   ============================================================ */
#define V13_D 128
#define V13_Q 1000
#define V13_K 10
#define V13_C 20
#define V13_N 200000
#define V13_NL 1024
#define V13_ITERS 10
static int v13_N,v13_NL,v13_mode,v13_np,v13_nt;
static float *v13_DBf,*v13_Qf,*v13_n2f,*v13_cent,*v13_cn2;
static uint8_t *v13_DBu,*v13_Qu,*v13_LU,*v13_C8;
static int *v13_n2i,*v13_ln2,*v13_order,*v13_asg,*v13_lstart,*v13_cn2i;
static int v13_R[V13_Q][V13_K],v13_GT[V13_Q][V13_K];
static atomic_int v13_qnext; static atomic_long v13_scan; static uint64_t v13_sd=88172645463325252ULL;
static double v13_rnd(void){v13_sd^=v13_sd<<13;v13_sd^=v13_sd>>7;v13_sd^=v13_sd<<17;return (v13_sd>>11)*(1.0/9007199254740992.0);}
static inline float v13_dotf(const float*a,const float*b){float acc[16]={0};for(int j=0;j<V13_D;j+=16)for(int l=0;l<16;l++)acc[l]+=a[j+l]*b[j+l];float r=0;for(int l=0;l<16;l++)r+=acc[l];return r;}
static inline int v13_dotu(const uint8_t*a,const uint8_t*b){uint32_t acc=0;for(int j=0;j<V13_D;j++)acc+=(uint32_t)a[j]*b[j];return (int)acc;}
static void v13_topk(const float*sc,const int*id,int n,int*out){float v[V13_K];for(int k=0;k<V13_K;k++){v[k]=-1e30f;out[k]=-1;}
 for(int i=0;i<n;i++){float x=sc[i];if(x<=v[V13_K-1])continue;int k=V13_K-1;while(k>0&&v[k-1]<x){v[k]=v[k-1];out[k]=out[k-1];k--;}v[k]=x;out[k]=id?id[i]:i;}}
#define V13_INS(S_,ID_) do{int _s=(S_),_k;if(n<V13_C)_k=n++;else if(_s>lcv[V13_C-1])_k=V13_C-1;else break;\
 while(_k>0&&lcv[_k-1]<_s){lcv[_k]=lcv[_k-1];lcand[_k]=lcand[_k-1];_k--;}lcv[_k]=_s;lcand[_k]=(ID_);}while(0)
static void*v13_worker(void*arg){(void)arg;float*Sl=v13_mode==0?malloc((size_t)v13_N*4):0;int lcv[V13_C],lcand[V13_C];float lcs[V13_C];
 int*csc=v13_mode==2?malloc((size_t)v13_NL*4):0;long sc=0;
 for(;;){int q=atomic_fetch_add(&v13_qnext,1);if(q>=V13_Q)break;const float*qf=v13_Qf+(size_t)q*V13_D;const uint8_t*qu=v13_Qu+(size_t)q*V13_D;
  if(v13_mode==0){for(int i=0;i<v13_N;i++)Sl[i]=v13_dotf(qf,v13_DBf+(size_t)i*V13_D)-0.5f*v13_n2f[i];v13_topk(Sl,0,v13_N,v13_R[q]);continue;}
  int n=0;
  if(v13_mode==1){for(int i=0;i<v13_N;i++)V13_INS(2*v13_dotu(qu,v13_DBu+(size_t)i*V13_D)-v13_n2i[i],i);}
  else{for(int c=0;c<v13_NL;c++)csc[c]=2*v13_dotu(qu,v13_C8+(size_t)c*V13_D)-v13_cn2i[c];
   int pl[128],pv[128];for(int k=0;k<v13_np;k++){pv[k]=INT32_MIN;pl[k]=-1;}
   for(int c=0;c<v13_NL;c++){int x=csc[c];if(x<=pv[v13_np-1])continue;int k=v13_np-1;while(k>0&&pv[k-1]<x){pv[k]=pv[k-1];pl[k]=pl[k-1];k--;}pv[k]=x;pl[k]=c;}
   for(int p=0;p<v13_np;p++){int l=pl[p];if(l<0)continue;for(int x=v13_lstart[l];x<v13_lstart[l+1];x++)V13_INS(2*v13_dotu(qu,v13_LU+(size_t)x*V13_D)-v13_ln2[x],v13_order[x]);sc+=v13_lstart[l+1]-v13_lstart[l];}}
  for(int c=0;c<n;c++)lcs[c]=v13_dotf(qf,v13_DBf+(size_t)lcand[c]*V13_D)-0.5f*v13_n2f[lcand[c]];v13_topk(lcs,lcand,n,v13_R[q]);}
 atomic_fetch_add(&v13_scan,sc);free(Sl);free(csc);return 0;}
static double v13_runmt(int mode,int np){v13_mode=mode;v13_np=np;atomic_store(&v13_qnext,0);atomic_store(&v13_scan,0);pthread_t th[16];double t0=now_s();
 for(int t=0;t<v13_nt;t++)pthread_create(&th[t],0,v13_worker,0);for(int t=0;t<v13_nt;t++)pthread_join(th[t],0);return now_s()-t0;}
static double v13_med3(int mode,int np){double w[3];for(int r=0;r<3;r++)w[r]=v13_runmt(mode,np);return v9_median3(w[0],w[1],w[2]);}
static double v13_recall(void){double h=0;for(int q=0;q<V13_Q;q++)for(int k=0;k<V13_K;k++)for(int g=0;g<V13_K;g++)if(v13_R[q][k]>=0&&v13_R[q][k]==v13_GT[q][g]){h++;break;}return h/(V13_Q*V13_K);}
typedef struct{int lo,hi;}v13_job_t;static const float*v13_Asrc;static int*v13_Aout;
static void*v13_ajob(void*p){v13_job_t*j=p;for(int i=j->lo;i<j->hi;i++){const float*x=v13_Asrc+(size_t)i*V13_D;int bc=0;float bs=-1e30f;
 for(int c=0;c<v13_NL;c++){float s=v13_dotf(v13_cent+(size_t)c*V13_D,x)-0.5f*v13_cn2[c];if(s>bs){bs=s;bc=c;}}v13_Aout[i]=bc;}return 0;}
static void v13_assign(const float*src,int n,int*out){v13_Asrc=src;v13_Aout=out;pthread_t th[16];v13_job_t jb[16];
 for(int t=0;t<v13_nt;t++){jb[t].lo=(int)((long)n*t/v13_nt);jb[t].hi=(int)((long)n*(t+1)/v13_nt);pthread_create(&th[t],0,v13_ajob,&jb[t]);}for(int t=0;t<v13_nt;t++)pthread_join(th[t],0);}
static void v13_upn2(void){for(int c=0;c<v13_NL;c++)v13_cn2[c]=v13_dotf(v13_cent+(size_t)c*V13_D,v13_cent+(size_t)c*V13_D);}
static int v13_readf(const char*p,float*o,int mx){FILE*f=fopen(p,"rb");if(!f)return -1;int d,n=0;
 while(n<mx&&fread(&d,4,1,f)==1){if(d!=V13_D){fclose(f);return -1;}if(fread(o+(size_t)n*V13_D,4,V13_D,f)!=V13_D)break;n++;}fclose(f);return n;}
static uint8_t v13_u8(float v){return (uint8_t)lrintf(v<0?0:v>255?255:v);}
static int v13_ivf_section(void){
 printf("\n=== §DELTA V13 IVF VECTOR SEARCH (SIFT REAL DATA) ===\n");
 long nc=sysconf(_SC_NPROCESSORS_ONLN);v13_nt=nc>1?(int)nc-1:1;if(v13_nt>16)v13_nt=16;
 v13_N=V13_N;v13_NL=V13_NL;v13_DBf=malloc((size_t)v13_N*V13_D*4);v13_Qf=malloc((size_t)V13_Q*V13_D*4);
 int nb=v13_readf("sift/sift_base.fvecs",v13_DBf,v13_N),nq=v13_readf("sift/sift_query.fvecs",v13_Qf,V13_Q);
 if(nb<=0||nq<V13_Q){printf("V13_SKIPPED=SIFT_FILES_MISSING\n");free(v13_DBf);free(v13_Qf);return 2;}
 v13_N=nb;v13_DBu=malloc((size_t)v13_N*V13_D);v13_Qu=malloc((size_t)V13_Q*V13_D);v13_LU=malloc((size_t)v13_N*V13_D);
 v13_n2f=malloc((size_t)v13_N*4);v13_n2i=malloc((size_t)v13_N*4);v13_ln2=malloc((size_t)v13_N*4);v13_order=malloc((size_t)v13_N*4);v13_asg=malloc((size_t)v13_N*4);
 for(int i=0;i<v13_N;i++){int s=0;for(int j=0;j<V13_D;j++){uint8_t u=v13_u8(v13_DBf[(size_t)i*V13_D+j]);v13_DBu[(size_t)i*V13_D+j]=u;s+=u*u;}v13_n2i[i]=s;v13_n2f[i]=(float)s;}
 for(int i=0;i<V13_Q*V13_D;i++)v13_Qu[i]=v13_u8(v13_Qf[i]);
 printf("V13_CONFIG N=%d D=%d Q=%d NLIST=%d THREADS=%d C=%d METRIC=L2\n",v13_N,V13_D,V13_Q,v13_NL,v13_nt,V13_C);
 double base=v13_runmt(0,0);memcpy(v13_GT,v13_R,sizeof(v13_GT));
 printf("V13_FULL_FLOAT32_MT WALL=%.3f s (GROUND_TRUTH)\n",base);
 double wu=v13_med3(1,0);printf("V13_FULL_UINT8_MT WALL_MED=%.3f s SPEEDUP=x%.2f RECALL@10=%.3f\n",wu,base/wu,v13_recall());
 double t0=now_s();int ts=30*v13_NL;if(ts<40000)ts=40000;if(ts>v13_N)ts=v13_N;
 float*T=malloc((size_t)ts*V13_D*4);int*ta=malloc((size_t)ts*4);double*sum=malloc((size_t)v13_NL*V13_D*8);int*cnt=malloc((size_t)v13_NL*4);
 v13_cent=malloc((size_t)v13_NL*V13_D*4);v13_cn2=malloc((size_t)v13_NL*4);v13_C8=malloc((size_t)v13_NL*V13_D);v13_cn2i=malloc((size_t)v13_NL*4);v13_lstart=calloc(v13_NL+1,4);
 for(int i=0;i<ts;i++){int k=(int)(v13_rnd()*v13_N);memcpy(T+(size_t)i*V13_D,v13_DBf+(size_t)k*V13_D,V13_D*4);}
 for(int c=0;c<v13_NL;c++)memcpy(v13_cent+(size_t)c*V13_D,T+(size_t)((int)(v13_rnd()*ts))*V13_D,V13_D*4);
 for(int it=0;it<V13_ITERS;it++){v13_upn2();v13_assign(T,ts,ta);memset(sum,0,(size_t)v13_NL*V13_D*8);memset(cnt,0,(size_t)v13_NL*4);
  for(int i=0;i<ts;i++){int c=ta[i];cnt[c]++;for(int j=0;j<V13_D;j++)sum[(size_t)c*V13_D+j]+=T[(size_t)i*V13_D+j];}
  for(int c=0;c<v13_NL;c++){if(cnt[c])for(int j=0;j<V13_D;j++)v13_cent[(size_t)c*V13_D+j]=(float)(sum[(size_t)c*V13_D+j]/cnt[c]);
   else memcpy(v13_cent+(size_t)c*V13_D,T+(size_t)((int)(v13_rnd()*ts))*V13_D,V13_D*4);}}
 v13_upn2();v13_assign(v13_DBf,v13_N,v13_asg);
 for(int c=0;c<v13_NL;c++){int s=0;for(int j=0;j<V13_D;j++){uint8_t u=v13_u8(v13_cent[(size_t)c*V13_D+j]);v13_C8[(size_t)c*V13_D+j]=u;s+=u*u;}v13_cn2i[c]=s;}
 for(int i=0;i<v13_N;i++)v13_lstart[v13_asg[i]+1]++;for(int c=0;c<v13_NL;c++)v13_lstart[c+1]+=v13_lstart[c];
 int*pos=malloc((size_t)(v13_NL+1)*4);memcpy(pos,v13_lstart,(size_t)(v13_NL+1)*4);
 for(int i=0;i<v13_N;i++){int x=pos[v13_asg[i]]++;v13_order[x]=i;memcpy(v13_LU+(size_t)x*V13_D,v13_DBu+(size_t)i*V13_D,V13_D);v13_ln2[x]=v13_n2i[i];}
 int mx=0;for(int c=0;c<v13_NL;c++){int s=v13_lstart[c+1]-v13_lstart[c];if(s>mx)mx=s;}
 printf("V13_KMEANS TRAIN=%d ITERS=%d BUILD_WALL=%.2f s LIST_MAX=%d IMBALANCE=x%.2f\n",ts,V13_ITERS,now_s()-t0,mx,mx/((double)v13_N/v13_NL));
 int nps[3]={16,32,64};double rec64=0;
 for(int i=0;i<3;i++){double w=v13_med3(2,nps[i]);double rec=v13_recall();if(nps[i]==64)rec64=rec;
  printf("V13_IVF NPROBE=%d SCANNED=%.2f%% WALL_MED=%.4f s SPEEDUP_VS_F32=x%.2f VS_U8FULL=x%.2f RECALL@10=%.3f\n",
   nps[i],100.0*atomic_load(&v13_scan)/((double)V13_Q*v13_N),w,base/w,wu/w,rec);}
 int ok=rec64>=0.98;printf("V13_VALIDATION=%s (RECALL@10 NPROBE64 >= 0.98)\n",ok?"OK":"FAIL");
 free(T);free(ta);free(sum);free(cnt);free(pos);free(v13_cent);free(v13_cn2);free(v13_C8);free(v13_cn2i);free(v13_lstart);
 free(v13_DBf);free(v13_Qf);free(v13_DBu);free(v13_Qu);free(v13_LU);free(v13_n2f);free(v13_n2i);free(v13_ln2);free(v13_order);free(v13_asg);
 return ok;}

'''
call="    v13_ivf_section();\n\n"
assert s.count(a)==1 and s.count(b)==1, "ANCRES"
assert "v13_ivf_section" not in s, "V13 DEJA PRESENT"
s=s.replace(a,fn+a).replace(b,call+b); open(f,"w").write(s); print("PATCH V13 OK")
