function [Hx,Hy,Hz,normH]=get_mag(mesh,solver,phy)
% Recover curlE from first-order basis, then H via frequency-domain Maxwell:
% H = 1/(i*k0*mu_r) * curlE
% i = imaginary unit; k0 = 2*pi/lambda; mu_r from phy.mur per mesh region.

Hx=zeros(mesh.NbrVertex,1);
Hy=zeros(mesh.NbrVertex,1);
Hz=zeros(mesh.NbrVertex,1);
nHx=zeros(mesh.NbrVertex,1);
nHy=zeros(mesh.NbrVertex,1);
nHz=zeros(mesh.NbrVertex,1);

u=[1,0,0,0];
v=[0,1,0,0];
w=[0,0,1,0];

k0=phy.k0;

for n=1:mesh.NbrTet
    %vertex
    x=mesh.Vertex(mesh.Tet(n,:),1);
    y=mesh.Vertex(mesh.Tet(n,:),2);
    z=mesh.Vertex(mesh.Tet(n,:),3);

    %jac
    Jac=zeros(3,3);
    Jac(1,1)=x(1)-x(4);Jac(1,2)=y(1)-y(4);Jac(1,3)=z(1)-z(4);
    Jac(2,1)=x(2)-x(4);Jac(2,2)=y(2)-y(4);Jac(2,3)=z(2)-z(4);
    Jac(3,1)=x(3)-x(4);Jac(3,2)=y(3)-y(4);Jac(3,3)=z(3)-z(4);
    % curlE transform (consistent with assembly_equ.m)
    TJac = (Jac')/det(Jac);

    mu_r = phy.mur(mesh.DomainOfTet(n));
    factor = - 1i /(k0*mu_r*120*pi);

    % curlE basis values at each vertex (curlE[:,edgeBasis,vertexPt])
    curlE=zeros(3,6,4);
    for i=1:4
        for j=1:6
            curlE(:,j,i)=getBF(2,j,u(i),v(i),w(i));
            curlE(:,j,i)=TJac*curlE(:,j,i);
        end
    end

    % Combine curlE basis with DOFs solver.x -> curlE at vertices
    for i=1:4
        vertId = mesh.Tet(n,i);
        for j=1:6
            edgeId = mesh.EdgeOfTet(n,j);
            Hx(vertId) = Hx(vertId) + factor*curlE(1,j,i)*solver.x(edgeId);
            Hy(vertId) = Hy(vertId) + factor*curlE(2,j,i)*solver.x(edgeId);
            Hz(vertId) = Hz(vertId) + factor*curlE(3,j,i)*solver.x(edgeId);
        end
        nHx(vertId)=nHx(vertId)+1;
        nHy(vertId)=nHy(vertId)+1;
        nHz(vertId)=nHz(vertId)+1;
    end
end

Hx=Hx./nHx;
Hy=Hy./nHy;
Hz=Hz./nHz;

normH=abs(Hx.*Hx + Hy.*Hy + Hz.*Hz);

end

