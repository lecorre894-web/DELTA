import re,sys
src="delta_ryzen_ddr5_dual_bench.c";dst="delta_ryzen_ddr5_dual_bench_v2.c"
s=open(src).read()
a=s.find("static void *memory_worker(void *arg)")
if a<0:sys.exit("ECHEC : memory_worker introuvable, rien ecrit")
b=s.find("return NULL;",a);b=s.find("}",b)+1
NEW='''static void *memory_worker(void *arg)
{
    /* V2 : meme travail exact (lit, ecrit x+1, somme), en une boucle continue.
       restrict = src et dst ne se chevauchent pas -> le compilateur vectorise (AVX). */
    worker_t *w = (worker_t *)arg;
    uint64_t sum = 0;
    uint8_t *a = w->src, *b = w->dst;
    const size_t n = (size_t)MEM_BYTES / 8;
    for (int pass=0; pass<PASSES; pass++) {
        const uint64_t *restrict s = (const uint64_t *)a;
        uint64_t *restrict d = (uint64_t *)b;
        for (size_t i=0; i<n; i++) { uint64_t x = s[i]; d[i] = x + 1; sum += x; }
        uint8_t *t = a; a = b; b = t;
    }
    w->src = a; w->dst = b;
    w->checksum = sum;
    return NULL;
}'''
s=s[:a]+NEW+s[b:]
s,k=re.subn(r"#define\s+ZONE_KIB\s+(\d+)",r"#ifndef ZONE_KIB\n#define ZONE_KIB \1\n#endif",s,count=1)
if not k:sys.exit("ECHEC : #define ZONE_KIB introuvable, rien ecrit")
open(dst,"w").write(s);print("V2_ECRIT :",dst,"(original intact)")
