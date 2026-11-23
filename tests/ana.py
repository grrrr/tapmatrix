import numpy as np
x=np.fromfile('out.f32',np.float32).reshape(-1,5).T.astype(float)
src=x[4]; N=len(src); n=np.arange(N, dtype=float)
# intended delay (samples) and gain, piecewise linear; ramps start at the block of the message
def pw(t0, segs):
    v=np.full(N, t0, float)
    for start, length, a, b in segs:
        k=np.arange(start, start+length); v[k]=a+(b-a)*(k-start)/length; v[start+length:]=b
    return v
D=pw(13.37*48, [(24000,9600,13.37*48,20.11*48),(48000,14400,20.11*48,5.5*48)])
G=pw(0.7, [(24000,9600,0.7,0.3),(48000,14400,0.3,1.0)])
def interp(D):
    t=n-D; j=np.floor(t).astype(int); fp=t-j; frac=1-fp
    c=src[j]; b=src[j+1]; a=src[np.minimum(j+2,N-1)]; d=src[j-1]; cb=c-b
    return b+frac*(cb-(1/6)*fp*((d-a-3*cb)*frac+(d+2*a-3*b)))
ref=G*interp(D)
def snr(r,y): e=y-r; return 10*np.log10(np.sum(r**2)/np.sum(e**2))
print(f"{'':36s}{'Pd vs ideal':>12s}{'tapmatrix vs ideal':>20s}{'tapmatrix vs Pd':>17s}")
segs=((2000,24000,'before ramp'),(24000,33600,'ramp 13.37->20.11 ms, gain .7->.3'),(33600,48000,'after'),(48000,62400,'ramp 20.11->5.5 ms, gain .3->1'),(62400,144000,'after'))
for a,b,nm in segs:
    print(f"{nm:36s}{snr(ref[a:b],x[2][a:b]):12.1f}{snr(ref[a:b],x[3][a:b]):20.1f}{snr(x[2][a:b],x[3][a:b]):17.1f}")
ref0=0.7*interp(np.full(N,13.37*48))
print(f"{'static tap':36s}{snr(ref0[2000:],x[0][2000:]):12.1f}{snr(ref0[2000:],x[1][2000:]):20.1f}{snr(x[0][2000:],x[1][2000:]):17.1f}")
