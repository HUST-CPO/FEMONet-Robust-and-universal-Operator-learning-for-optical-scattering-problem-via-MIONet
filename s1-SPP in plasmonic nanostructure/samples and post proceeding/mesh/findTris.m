function EdgeIndex=findTris(targetflag1,Mesh)
index=find(Mesh.BedgeID==targetflag1(1,1));
for i=2:size(targetflag1,2)
    index=[index;find(Mesh.BedgeID==targetflag1(1,i))];
end
index=sort(index);
Boundary1=sort(Mesh.Bedge(index,:),2);
[~ ,PBCEdge] = ismember(Boundary1,Mesh.edges,'rows');
EdgeIndex=sort(PBCEdge);