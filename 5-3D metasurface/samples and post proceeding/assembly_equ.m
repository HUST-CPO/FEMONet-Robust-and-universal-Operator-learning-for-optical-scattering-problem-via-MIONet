function solver=assembly_equ(phy,mesh,solver)

solver.dof=mesh.NbrEdge;
k0=2*pi/phy.lda0;

% Tetrahedron Gauss points
[u,v,w,weight,nbrGP]=getGaussPoints(1);

Ai=zeros(mesh.NbrTet*36,1);
Aj=zeros(mesh.NbrTet*36,1);
Av=zeros(mesh.NbrTet*36,1);
for n=1:mesh.NbrTet
    %vertex
    x=mesh.Vertex(mesh.Tet(n,:),1);
    y=mesh.Vertex(mesh.Tet(n,:),2);
    z=mesh.Vertex(mesh.Tet(n,:),3);

    %length of edge
    l=zeros(6,1);
    l(1)=sqrt((x(1)-x(2))^2+(y(1)-y(2))^2+(z(1)-z(2))^2);
    l(2)=sqrt((x(1)-x(3))^2+(y(1)-y(3))^2+(z(1)-z(3))^2);
    l(3)=sqrt((x(1)-x(4))^2+(y(1)-y(4))^2+(z(1)-z(4))^2);
    l(4)=sqrt((x(2)-x(3))^2+(y(2)-y(3))^2+(z(2)-z(3))^2);
    l(5)=sqrt((x(2)-x(4))^2+(y(2)-y(4))^2+(z(2)-z(4))^2);
    l(6)=sqrt((x(3)-x(4))^2+(y(3)-y(4))^2+(z(3)-z(4))^2);

    %jac
    Jac=zeros(3,3);
    Jac(1,1)=x(1)-x(4);Jac(1,2)=y(1)-y(4);Jac(1,3)=z(1)-z(4);
    Jac(2,1)=x(2)-x(4);Jac(2,2)=y(2)-y(4);Jac(2,3)=z(2)-z(4);
    Jac(3,1)=x(3)-x(4);Jac(3,2)=y(3)-y(4);Jac(3,3)=z(3)-z(4);
    DetJac=abs(det(Jac));
    TJac=Jac'/det(Jac);

    %BF
    E=zeros(3,6,nbrGP);curlE=zeros(3,6,nbrGP);
    for i=1:nbrGP
        for j=1:6
            E(:,j,i)=getBF(1,j,u(i),v(i),w(i));
            E(:,j,i)=Jac\E(:,j,i)*l(j);
            curlE(:,j,i)=getBF(2,j,u(i),v(i),w(i));
            curlE(:,j,i)=TJac*curlE(:,j,i)*l(j);
        end
    end

    %matrix
    mur=phy.mur(mesh.DomainOfTet(n));
    eps=phy.eps(mesh.DomainOfTet(n));

    %submatrix
    Ae=zeros(6,6);
    for i=1:6
        for j=1:6
            for k=1:nbrGP
                Ae(i,j)=Ae(i,j)+weight(k)*DetJac*sum(curlE(:,i,k).*curlE(:,j,k))...
                    -weight(k)*DetJac*k0*k0*sum(E(:,i,k).*(eps*E(:,j,k)));
            end
        end
    end

    %put in matrix
    for i=1:6
        for j=1:6
            index=(n-1)*36+(i-1)*6+j;
            Ai(index)=mesh.EdgeOfTet(n,i);
            Aj(index)=mesh.EdgeOfTet(n,j);
            Av(index)=Ae(i,j);
        end
    end
end

solver.Ai=[solver.Ai;Ai];
solver.Aj=[solver.Aj;Aj];
solver.Av=[solver.Av;Av];

end