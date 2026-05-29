function [copyOfBedge,EdgePhi]=GetcopyOfBedge(PBCEdge1,PBCEdge2,Mesh)

dif0=[0.942622950819672,0;0.942622950819672,0];
rows = [1,2];

for i = 1:size(PBCEdge1,1)
    for j = 1:size(PBCEdge2,1)
        dif = Mesh.vertex(Mesh.edges(PBCEdge2(j),:),:)-Mesh.vertex(Mesh.edges(PBCEdge1(i),:),:);
        if(all(all(abs(dif(rows,:)-dif0)<0.0001)))
            copyOfBedge(i,1)=PBCEdge1(i);
            copyOfBedge(i,2)=PBCEdge2(j);
            EdgePhi(i,1) = 1;
        end
        dif = Mesh.vertex(Mesh.edges(PBCEdge2(j),:),:)-flipud(Mesh.vertex(Mesh.edges(PBCEdge1(i),:),:));
        if(all(all(abs(dif(rows,:)-dif0)<0.0001)))
            copyOfBedge(i,1)=PBCEdge1(i);
            copyOfBedge(i,2)=PBCEdge2(j);
            EdgePhi(i,1) = -1;
        end
    end
end