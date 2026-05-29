function edgeIndex=FindEdge(flag,mesh)
%n*2 1-tri 2-number edge of tri 

if isempty(flag)
    edgeIndex=[];
    return
end

index=find(mesh.BedgeID==flag(1));
if length(flag)>1
    for i=2:length(flag)
        index=[index;find(mesh.BedgeID==flag(i))];
    end
end
index=sort(index);
boundary=mesh.Bedge(index,:);
boundary=sort(boundary,2);

[~,index]=ismember(boundary,mesh.edges,'rows');

edgeOfTri=reshape(mesh.edgesOfTri',length(mesh.edgesOfTri)*3,1);
[~,index]=ismember(index,edgeOfTri,'rows');
n=length(index);
edgeIndex=zeros(n,2);
edgeIndex(:,1)=fix((index-1)/3)+1;
edgeIndex(:,2)=index-(edgeIndex(:,1)-1)*3;

end