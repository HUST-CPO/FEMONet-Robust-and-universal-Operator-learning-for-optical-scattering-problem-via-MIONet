function solver=assembly_out(phy,mesh,solver)
k0=2*pi/phy.lda0;
% Triangle surface quadrature
[u,v,~,weight,nbrGP]=getGaussPoints(2);
w=1-u-v;

nbrOut=length(mesh.outIndex);
Ai=zeros(nbrOut*9,1);
Aj=zeros(nbrOut*9,1);
Av=zeros(nbrOut*9,1);
%loop of all tri
for n=1:nbrOut
    domain=mesh.DomainOfTri(mesh.outIndex(n));
    numTet=mesh.ConnOfTri(mesh.outIndex(n),1);
    numFace=mesh.ConnOfTri(mesh.outIndex(n),2);

    %vertex
    x=mesh.Vertex(mesh.Tet(numTet,:),1);
    y=mesh.Vertex(mesh.Tet(numTet,:),2);
    z=mesh.Vertex(mesh.Tet(numTet,:),3);

    %length of edge
    l=zeros(6,1);
    l(1)=sqrt((x(1)-x(2))^2+(y(1)-y(2))^2+(z(1)-z(2))^2);
    l(2)=sqrt((x(1)-x(3))^2+(y(1)-y(3))^2+(z(1)-z(3))^2);
    l(3)=sqrt((x(1)-x(4))^2+(y(1)-y(4))^2+(z(1)-z(4))^2);
    l(4)=sqrt((x(2)-x(3))^2+(y(2)-y(3))^2+(z(2)-z(3))^2);
    l(5)=sqrt((x(2)-x(4))^2+(y(2)-y(4))^2+(z(2)-z(4))^2);
    l(6)=sqrt((x(3)-x(4))^2+(y(3)-y(4))^2+(z(3)-z(4))^2);

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

    %normal
    normal=mesh.NormOfFace(domain,:)';
    
    %bf
    E=zeros(3,3,nbrGP); % 3 x 3 basis functions x nbrGP Gauss points
    for i=1:nbrGP
        u2=x2(1)*u(i)+x2(2)*v(i)+x2(3)*w(i);
        v2=y2(1)*u(i)+y2(2)*v(i)+y2(3)*w(i);
        w2=z2(1)*u(i)+z2(2)*v(i)+z2(3)*w(i);
        for j=1:3
            E(:,j,i)=getBF(1,index(j),u2,v2,w2);
            E(:,j,i)=Jac\E(:,j,i)*l(index(j));
        end
    end

    %mat
    eps=phy.eps(mesh.DomainOfTet(numTet));
    nn=sqrt(eps);

    %submatrix
    Ae=zeros(3,3);
    for i=1:3
        for j=1:3
            for k=1:nbrGP
                Ae(i,j)=Ae(i,j)+1i*k0*nn*integCoe*weight(k)*sum(E(:,i,k).*cross(normal,cross(E(:,j,k),normal)))*2;
            end
        end
    end

    %put in matrix
    for i=1:3
        for j=1:3
            index=(n-1)*9+(i-1)*3+j;
            Ai(index)=MappingIndex(i);
            Aj(index)=MappingIndex(j);
            Av(index)=Ae(i,j);
        end
    end
end

solver.Ai=[solver.Ai;Ai];
solver.Aj=[solver.Aj;Aj];
solver.Av=[solver.Av;Av];

end