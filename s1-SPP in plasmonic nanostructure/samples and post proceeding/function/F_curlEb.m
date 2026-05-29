function out=F_curlEb(x,y,physic)
out=[0;0;-1i*physic.k0*exp(1i*physic.k0*y)];