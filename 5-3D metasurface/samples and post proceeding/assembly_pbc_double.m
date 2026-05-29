function [mesh,solver]=assembly_pbc_double(mesh,solver)

% Find duplicate values in src and dst
tempSrc=sort([mesh.PBCIndex(:,1);mesh.PBCIndex2(:,1)]);
tbl = tabulate(tempSrc);% 3-column table: value, count, ...
ovSrcIndex=find(tbl(:,2)>1);
tempDst=sort([mesh.PBCIndex(:,2);mesh.PBCIndex2(:,2)]);
tbl2 = tabulate(tempDst);% 3-column table: value, count, ...
ovDstIndex=find(tbl2(:,2)>1);

% Final PBCIndex count
nbr1=length(ovSrcIndex);
nbr2=length(mesh.PBCIndex);
nbr3=length(mesh.PBCIndex2);
nbr=nbr2+nbr3-nbr1;
PBCIndex=zeros(nbr,3);


% In dst2, map ovDstIndex to src2/dst1
[~,index1]=ismember(ovDstIndex,mesh.PBCIndex2(:,2));% rows removed from PBC2
index2=mesh.PBCIndex2(index1,1);% src2/dst1
% In dst1, find src2/dst1
[~,index3]=ismember(index2,mesh.PBCIndex(:,2));% rows removed from PBC1
index4=mesh.PBCIndex(index3,1);% src1
% Store double-Bloch node pairs
PBCIndex(1:nbr1,1)=index4;
PBCIndex(1:nbr1,2)=ovDstIndex;
PBCIndex(1:nbr1,3)=mesh.PBCIndex2(index1,3).*mesh.PBCIndex(index3,3);

% Append remaining PBC1 pairs
tempindex=1:nbr2;
tempindex(index3)=[];
PBCIndex(nbr1+1:nbr2,:)=mesh.PBCIndex(tempindex,:);
tempindex=1:nbr3;
tempindex(index1)=[];
PBCIndex(nbr2+1:end,:)=mesh.PBCIndex2(tempindex,:);

P=speye(solver.dof);
phi=PBCIndex(:,3);
phi(1:nbr1)=mesh.PBCphi*mesh.PBCphi2.*phi(1:nbr1);
phi(nbr1+1:nbr2)=mesh.PBCphi.*phi(nbr1+1:nbr2);
phi(nbr2+1:end,:)=mesh.PBCphi2.*phi(nbr2+1:end,:);
nbrPBC=length(phi);
for i=1:nbrPBC
    P(PBCIndex(i,2),PBCIndex(i,1))=phi(i);
end
P(:,PBCIndex(:,2))=[];

solver.P=P;
mesh.Index=1:(mesh.NbrEdge);
mesh.Index(PBCIndex(:,2))=[];

end