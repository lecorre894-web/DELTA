import zlib,json
def _uv(x,o):
    while True:
        b=x&127;x>>=7
        if x:o.append(b|128)
        else:o.append(b);return
def _rv(d,i):
    x=s=0
    while True:
        b=d[i];i+=1;x|=(b&127)<<s;s+=7
        if b<128:return x,i
def pack(counts):
    n=len(next(iter(counts)));it=sorted((int(k,2),v) for k,v in counts.items());o=bytearray();_uv(n,o);_uv(len(it),o);p=0
    for i,v in it:_uv(i-p,o);_uv(v,o);p=i
    z=zlib.compress(bytes(o),9);return b"\x02"+z if len(z)<len(o) else b"\x01"+bytes(o)
def unpack(v):
    if isinstance(v,str):return json.loads(v)
    d=v[1:] if v[:1]==b"\x01" else zlib.decompress(v[1:]);n,i=_rv(d,0);k,i=_rv(d,i);out={};p=0
    for _ in range(k):
        dl,i=_rv(d,i);c,i=_rv(d,i);p+=dl;out[format(p,"0%db"%n)]=c
    return out
