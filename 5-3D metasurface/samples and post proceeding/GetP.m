function p=GetP(triIndex,mesh,phy,solver,s)

k0=2*pi/phy.lda0;
% Triangle surface quadrature
[u,v,~,weight,nbrGP]=getGaussPoints(2);
w=1-u-v;

p=0;
nbr=length(triIndex);
for n=1:nbr
    numTet=mesh.ConnOfTri(triIndex(n),1);
    numFace=mesh.ConnOfTri(triIndex(n),2);

    %vertex
    x=mesh.Vertex(mesh.Tet(numTet,:),1);
    y=mesh.Vertex(mesh.Tet(numTet,:),2);
    z=mesh.Vertex(mesh.Tet(numTet,:),3);

    MappingIndex=zeros(3,1);
    if numFace==1 %123
        x2=[1;0;0];
        y2=[0;1;0];
        z2=[0;0;1];
        x3=[x(1);x(2);x(3)];
        y3=[y(1);y(2);y(3)];
        z3=[z(1);z(2);z(3)];
        index=[1;2;4];
        MappingIndex(1)=mesh.EdgeOfTet(numTet,1);
        MappingIndex(2)=mesh.EdgeOfTet(numTet,2);
        MappingIndex(3)=mesh.EdgeOfTet(numTet,4);
    elseif numFace==2 %124
        x2=[1;0;0];
        y2=[0;1;0];
        z2=[0;0;0];
        x3=[x(1);x(2);x(4)];
        y3=[y(1);y(2);y(4)];
        z3=[z(1);z(2);z(4)];
        index=[1;3;5];
        MappingIndex(1)=mesh.EdgeOfTet(numTet,1);
        MappingIndex(2)=mesh.EdgeOfTet(numTet,3);
        MappingIndex(3)=mesh.EdgeOfTet(numTet,5);
    elseif numFace==3 %134
        x2=[1;0;0];
        y2=[0;0;0];
        z2=[0;1;0];
        x3=[x(1);x(3);x(4)];
        y3=[y(1);y(3);y(4)];
        z3=[z(1);z(3);z(4)];
        index=[2;3;6];
        MappingIndex(1)=mesh.EdgeOfTet(numTet,2);
        MappingIndex(2)=mesh.EdgeOfTet(numTet,3);
        MappingIndex(3)=mesh.EdgeOfTet(numTet,6);
    elseif numFace==4 %234
        x2=[0;0;0];
        y2=[1;0;0];
        z2=[0;1;0];
        x3=[x(2);x(3);x(4)];
        y3=[y(2);y(3);y(4)];
        z3=[z(2);z(3);z(4)];
        index=[4;5;6];
        MappingIndex(1)=mesh.EdgeOfTet(numTet,4);
        MappingIndex(2)=mesh.EdgeOfTet(numTet,5);
        MappingIndex(3)=mesh.EdgeOfTet(numTet,6);
    end

    %jac
    Jac=zeros(3,3);
    Jac(1,1)=x(1)-x(4);Jac(1,2)=y(1)-y(4);Jac(1,3)=z(1)-z(4);
    Jac(2,1)=x(2)-x(4);Jac(2,2)=y(2)-y(4);Jac(2,3)=z(2)-z(4);
    Jac(3,1)=x(3)-x(4);Jac(3,2)=y(3)-y(4);Jac(3,3)=z(3)-z(4);

    %int coe
    a=sqrt((x3(1)-x3(2))^2+(y3(1)-y3(2))^2+(z3(1)-z3(2))^2);
    b=sqrt((x3(1)-x3(3))^2+(y3(1)-y3(3))^2+(z3(1)-z3(3))^2);
    c=sqrt((x3(2)-x3(3))^2+(y3(2)-y3(3))^2+(z3(2)-z3(3))^2);
    integCoe = 0.25 * sqrt((a + b + c) * (a + b - c) * (a - b + c) * (b + c - a));

    p=p+integCoe*s(n);

    

end