function [Ex,Ey,Ez,normE]=get_ele(mesh,solver)

Ex=zeros(mesh.NbrVertex,1);
Ey=zeros(mesh.NbrVertex,1);
Ez=zeros(mesh.NbrVertex,1);
nEx=zeros(mesh.NbrVertex,1);
nEy=zeros(mesh.NbrVertex,1);
nEz=zeros(mesh.NbrVertex,1);

u=[1,0,0,0];
v=[0,1,0,0];
w=[0,0,1,0];

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

    %BF
    E=zeros(3,6,4);
    for i=1:4
        for j=1:6
            E(:,j,i)=getBF(1,j,u(i),v(i),w(i));
            E(:,j,i)=Jac\E(:,j,i)*l(j);
        end
    end

    % Electric field
    for j=1:4
        for i=1:6
            Ex(mesh.Tet(n,j))=Ex(mesh.Tet(n,j))+E(1,i,j)*solver.x(mesh.EdgeOfTet(n,i));
            Ey(mesh.Tet(n,j))=Ey(mesh.Tet(n,j))+E(2,i,j)*solver.x(mesh.EdgeOfTet(n,i));
            Ez(mesh.Tet(n,j))=Ez(mesh.Tet(n,j))+E(3,i,j)*solver.x(mesh.EdgeOfTet(n,i));
        end
        nEx(mesh.Tet(n,j))=nEx(mesh.Tet(n,j))+1;
        nEy(mesh.Tet(n,j))=nEy(mesh.Tet(n,j))+1;
        nEz(mesh.Tet(n,j))=nEz(mesh.Tet(n,j))+1;
    end
end

Ex=Ex./nEx;
Ey=Ey./nEy;
Ez=Ez./nEz;
normE=abs(Ex.*Ex+Ey.*Ey+Ez.*Ez);

end