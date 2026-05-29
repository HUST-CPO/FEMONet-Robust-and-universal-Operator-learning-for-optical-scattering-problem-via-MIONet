function [Src,Dst,value]=GetPeriodicBoundIndex(mesh)

SrcBoundaryFlag=mesh.SrcID;
DstBoundaryFlag=mesh.DstID;
%src
index=find(mesh.BedgeID==SrcBoundaryFlag(1));
for i=2:length(SrcBoundaryFlag)
    index=[index;find(mesh.BedgeID==SrcBoundaryFlag(i))];
end
index=sort(index);
SrcBoundary=mesh.Bedge(index,:);
SrcBoundary=sort(SrcBoundary,2);

%dst
index=find(mesh.BedgeID==DstBoundaryFlag(1));
for i=2:length(DstBoundaryFlag)
    index=[index;find(mesh.BedgeID==DstBoundaryFlag(i))];
end
index=sort(index);
DstBoundary=mesh.Bedge(index,:);
DstBoundary=sort(DstBoundary,2);

%find Index
SrcIndex=zeros(length(SrcBoundary),1);
DstIndex=zeros(length(DstBoundary),1);
for i=1:length(SrcBoundary)
    SrcIndex(i)=find(ismember(mesh.edges,SrcBoundary(i,:),'rows'));
    DstIndex(i)=find(ismember(mesh.edges,DstBoundary(i,:),'rows'));
end

SrcValue=zeros(length(SrcBoundary),2);
DstValue=zeros(length(SrcBoundary),2);
for i=1:length(SrcBoundary)
    x1=mesh.vertex(SrcBoundary(i,2),1);
    y1=mesh.vertex(SrcBoundary(i,2),2);
    x2=mesh.vertex(SrcBoundary(i,1),1);
    y2=mesh.vertex(SrcBoundary(i,1),2);
    SrcValue(i,1)=sqrt(((x1+x2)/2)^2+((y1+y2)/2)^2);
    if x1>x2
        SrcValue(i,2)=1;
    else
        SrcValue(i,2)=-1;
    end

    x3=mesh.vertex(DstBoundary(i,2),1);
    y3=mesh.vertex(DstBoundary(i,2),2);
    x4=mesh.vertex(DstBoundary(i,1),1);
    y4=mesh.vertex(DstBoundary(i,1),2);
    DstValue(i,1)=sqrt(((x3+x4)/2)^2+((y3+y4)/2)^2);
    if x4>x3
        DstValue(i,2)=1;
    else
        DstValue(i,2)=-1;
    end
end

Src=zeros(length(SrcBoundary),1);
Dst=zeros(length(DstBoundary),1);
value=zeros(length(DstBoundary),1);
for i=1:length(SrcBoundary)
    Src(i)=SrcIndex(i);
    for j=1:length(SrcBoundary)
        if (abs(SrcValue(i,1)-DstValue(j,1))<1e-8)
            Dst(i)=DstIndex(j);
            if (SrcValue(i,2)==DstValue(j,2))
                value(i)=1;
            else
                value(i)=-1;
            end
            break;
        end
    end
end
