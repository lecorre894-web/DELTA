import numpy as np,torch
from qiskit import transpile
DEV=torch.device("cuda")
def _u(s,q,m):
    v=s.view(-1,2,2**q);a=v[:,0].clone();b=v[:,1].clone();v[:,0]=m[0,0]*a+m[0,1]*b;v[:,1]=m[1,0]*a+m[1,1]*b
def _cx(s,n,c,t):
    if c>t:v=s.view(2**(n-c-1),2,2**(c-t-1),2,2**t);x=v[:,1,:,0,:].clone();v[:,1,:,0,:]=v[:,1,:,1,:];v[:,1,:,1,:]=x
    else:v=s.view(2**(n-t-1),2,2**(t-c-1),2,2**c);x=v[:,0,:,1,:].clone();v[:,0,:,1,:]=v[:,1,:,1,:];v[:,1,:,1,:]=x
def sim_gpu(qc,shots):
    n=qc.num_qubits;t=transpile(qc.remove_final_measurements(inplace=False),basis_gates=["u","cx"],optimization_level=0)
    dt=torch.complex128 if n<=27 else torch.complex64;s=torch.zeros(2**n,dtype=dt,device=DEV);s[0]=1
    for ins in t.data:
        q=[t.find_bit(b).index for b in ins.qubits];nm=ins.operation.name
        if nm=="cx":_cx(s,n,q[0],q[1])
        elif nm=="u":_u(s,q[0],torch.as_tensor(np.asarray(ins.operation.to_matrix()),dtype=dt,device=DEV))
    p=(s.abs()**2).double().cpu().numpy();del s;torch.cuda.empty_cache()
    p/=p.sum();rng=np.random.default_rng(1234);idx=rng.choice(p.size,size=shots,p=p);u,cn=np.unique(idx,return_counts=True)
    return {format(int(i),"0%db"%n):int(c) for i,c in zip(u,cn)},{"engine":"DELTA_STATEVECTOR_GPU","qubits":n,"device":torch.cuda.get_device_name(0),"dtype":str(dt).split(".")[-1]}
