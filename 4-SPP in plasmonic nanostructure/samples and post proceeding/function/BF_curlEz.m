function [out1,out2]=BF_curlEz(num,u,v)
if num==1
	out1=-1;
	out2=1; 
elseif num==2
	out1=0;
	out2=-1;
elseif num==3
	out1=1;
	out2=0;
end