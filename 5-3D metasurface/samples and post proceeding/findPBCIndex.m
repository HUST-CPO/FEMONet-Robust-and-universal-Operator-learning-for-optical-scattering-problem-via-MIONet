function PBCIndex=findPBCIndex(src,dst,dis,mesh)
%src+dis=dst
%PBCIndex n*3  1-edgeIndedxOfsrc 2-edgeIndexOfdst 3-+-1

% Find matching edge indices for src and dst
tempSrcTriIndex=findTri(src,mesh);% mesh.tri indices
tempDstTriIndex=findTri(dst,mesh);
tempSrcTriCoon=mesh.ConnOfTri(tempSrcTriIndex,:);% mesh.ConnOfTri data
tempDstTriCoon=mesh.ConnOfTri(tempDstTriIndex,:);
tempNbr=length(tempSrcTriCoon);
tempSrcEdge=zeros(tempNbr*3,1);% all mesh.Edge indices on src
tempDstEdge=zeros(tempNbr*3,1);
for n=1:tempNbr
    if tempSrcTriCoon(n,2)==1 %123 
        tempSrcEdge((n-1)*3+1)=mesh.EdgeOfTet(tempSrcTriCoon(n,1),1);
        tempSrcEdge((n-1)*3+2)=mesh.EdgeOfTet(tempSrcTriCoon(n,1),2);
        tempSrcEdge((n-1)*3+3)=mesh.EdgeOfTet(tempSrcTriCoon(n,1),4);
    elseif tempSrcTriCoon(n,2)==2
        tempSrcEdge((n-1)*3+1)=mesh.EdgeOfTet(tempSrcTriCoon(n,1),1);
        tempSrcEdge((n-1)*3+2)=mesh.EdgeOfTet(tempSrcTriCoon(n,1),3);
        tempSrcEdge((n-1)*3+3)=mesh.EdgeOfTet(tempSrcTriCoon(n,1),5);
    elseif tempSrcTriCoon(n,2)==3
        tempSrcEdge((n-1)*3+1)=mesh.EdgeOfTet(tempSrcTriCoon(n,1),2);
        tempSrcEdge((n-1)*3+2)=mesh.EdgeOfTet(tempSrcTriCoon(n,1),3);
        tempSrcEdge((n-1)*3+3)=mesh.EdgeOfTet(tempSrcTriCoon(n,1),6);
    elseif tempSrcTriCoon(n,2)==4
        tempSrcEdge((n-1)*3+1)=mesh.EdgeOfTet(tempSrcTriCoon(n,1),4);
        tempSrcEdge((n-1)*3+2)=mesh.EdgeOfTet(tempSrcTriCoon(n,1),5);
        tempSrcEdge((n-1)*3+3)=mesh.EdgeOfTet(tempSrcTriCoon(n,1),6);
    end

    if tempDstTriCoon(n,2)==1 %123 
        tempDstEdge((n-1)*3+1)=mesh.EdgeOfTet(tempDstTriCoon(n,1),1);
        tempDstEdge((n-1)*3+2)=mesh.EdgeOfTet(tempDstTriCoon(n,1),2);
        tempDstEdge((n-1)*3+3)=mesh.EdgeOfTet(tempDstTriCoon(n,1),4);
    elseif tempDstTriCoon(n,2)==2
        tempDstEdge((n-1)*3+1)=mesh.EdgeOfTet(tempDstTriCoon(n,1),1);
        tempDstEdge((n-1)*3+2)=mesh.EdgeOfTet(tempDstTriCoon(n,1),3);
        tempDstEdge((n-1)*3+3)=mesh.EdgeOfTet(tempDstTriCoon(n,1),5);
    elseif tempDstTriCoon(n,2)==3
        tempDstEdge((n-1)*3+1)=mesh.EdgeOfTet(tempDstTriCoon(n,1),2);
        tempDstEdge((n-1)*3+2)=mesh.EdgeOfTet(tempDstTriCoon(n,1),3);
        tempDstEdge((n-1)*3+3)=mesh.EdgeOfTet(tempDstTriCoon(n,1),6);
    elseif tempDstTriCoon(n,2)==4
        tempDstEdge((n-1)*3+1)=mesh.EdgeOfTet(tempDstTriCoon(n,1),4);
        tempDstEdge((n-1)*3+2)=mesh.EdgeOfTet(tempDstTriCoon(n,1),5);
        tempDstEdge((n-1)*3+3)=mesh.EdgeOfTet(tempDstTriCoon(n,1),6);
    end
end
% Deduplicate
[SrcEdge,~,~]=unique(tempSrcEdge);% src mesh.Edge indices
[DstEdge,~,~]=unique(tempDstEdge);

% Match src/dst edge pairs
nbrPBC=length(SrcEdge);
PBCIndex=zeros(nbrPBC,3);
PBCIndex(:,1)=SrcEdge;
% Nested loop: compare edge separation distances
dl=norm(dis);% target offset magnitude
err=dl*0.00005;
for i=1:nbrPBC
    for j=1:nbrPBC
        % Edge endpoint coordinates
        vertex1=mesh.Vertex(mesh.Edge(SrcEdge(i),1),:);
        vertex2=mesh.Vertex(mesh.Edge(SrcEdge(i),2),:);
        vertex3=mesh.Vertex(mesh.Edge(DstEdge(j),1),:);
        vertex4=mesh.Vertex(mesh.Edge(DstEdge(j),2),:);
        % Separation distances between edge pairs
        l1=abs(norm(vertex1-vertex3)-dl);
        l2=abs(norm(vertex2-vertex4)-dl);
        l3=abs(norm(vertex1-vertex4)-dl);
        l4=abs(norm(vertex2-vertex3)-dl);

        if (l1+l2)<err
            PBCIndex(i,2)=DstEdge(j);
            PBCIndex(i,3)=1;
            break;
        elseif (l3+l4)<err
            PBCIndex(i,2)=DstEdge(j);
            PBCIndex(i,3)=-1;
            break;
        else
            continue;
        end
    end
end

stop=0;

end