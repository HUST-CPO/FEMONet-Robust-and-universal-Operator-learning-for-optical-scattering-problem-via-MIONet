function solver=assembly_pbc_single(mesh,solver)

P=speye(solver.dof);
phi=mesh.PBCIndex(:,3)*mesh.PBCphi;
nbrPBC=length(phi);
for i=1:nbrPBC
    P(mesh.PBCIndex(i,2),mesh.PBCIndex(i,1))=phi(i);
end
P(:,mesh.PBCIndex(:,2))=[];

solver.P=P;

end