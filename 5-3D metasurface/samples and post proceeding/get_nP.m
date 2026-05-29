function nP=get_nP(triIndex,mesh,solver,k0)

[u,v,~,weight,nbrGP]=getGaussPoints(2);
w=1-u-v;

nbr=length(triIndex);
nP=0;

for n=1:nbr
    numTet=mesh.ConnOfTri(triIndex(n),1);
    numFace=mesh.ConnOfTri(triIndex(n),2);

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

    if numFace==1 %123
        x2=[1;0;0];
        y2=[0;1;0];
        z2=[0;0;1];
        x3=[x(1);x(2);x(3)];
        y3=[y(1);y(2);y(3)];
        z3=[z(1);z(2);z(3)];
    elseif numFace==2 %124
        x2=[1;0;0];
        y2=[0;1;0];
        z2=[0;0;0];
        x3=[x(1);x(2);x(4)];
        y3=[y(1);y(2);y(4)];
        z3=[z(1);z(2);z(4)];
    elseif numFace==3 %134
        x2=[1;0;0];
        y2=[0;0;0];
        z2=[0;1;0];
        x3=[x(1);x(3);x(4)];
        y3=[y(1);y(3);y(4)];
        z3=[z(1);z(3);z(4)];
    elseif numFace==4 %234
        x2=[0;0;0];
        y2=[1;0;0];
        z2=[0;1;0];
        x3=[x(2);x(3);x(4)];
        y3=[y(2);y(3);y(4)];
        z3=[z(2);z(3);z(4)];
    end

    %jac
    Jac=zeros(3,3);
    Jac(1,1)=x(1)-x(4);Jac(1,2)=y(1)-y(4);Jac(1,3)=z(1)-z(4);
    Jac(2,1)=x(2)-x(4);Jac(2,2)=y(2)-y(4);Jac(2,3)=z(2)-z(4);
    Jac(3,1)=x(3)-x(4);Jac(3,2)=y(3)-y(4);Jac(3,3)=z(3)-z(4);

    TJac = Jac'/det(Jac);
    factor = 1i /(k0*120*pi);

    %int coe
    a=sqrt((x3(1)-x3(2))^2+(y3(1)-y3(2))^2+(z3(1)-z3(2))^2);
    b=sqrt((x3(1)-x3(3))^2+(y3(1)-y3(3))^2+(z3(1)-z3(3))^2);
    c=sqrt((x3(2)-x3(3))^2+(y3(2)-y3(3))^2+(z3(2)-z3(3))^2);
    integCoe = 0.5 * sqrt((a + b + c) * (a + b - c) * (a - b + c) * (b + c - a));

    %bf
    E=zeros(3,6,nbrGP);curlE=zeros(3,6,nbrGP); % 3 x 6 basis x nbrGP Gauss points
    for i=1:nbrGP
        u2=x2(1)*u(i)+x2(2)*v(i)+x2(3)*w(i);
        v2=y2(1)*u(i)+y2(2)*v(i)+y2(3)*w(i);
        w2=z2(1)*u(i)+z2(2)*v(i)+z2(3)*w(i);
        for j=1:6
            E(:,j,i)=getBF(1,j,u2,v2,w2);
            E(:,j,i)=Jac\E(:,j,i)*l(j);
            curlE(:,j,i)=getBF(2,j,u2,v2,w2);
            curlE(:,j,i)=TJac*curlE(:,j,i)*l(j);
        end
    end


    % E and H at Gauss points
    Ex=zeros(nbrGP);Ey=zeros(nbrGP);Ez=zeros(nbrGP);
    Hx=zeros(nbrGP);Hy=zeros(nbrGP);Hz=zeros(nbrGP);
    for i=1:nbrGP
        for j=1:6
            Ex(i)=Ex(i)+E(1,j,i)*solver.x(mesh.EdgeOfTet(numTet,j));
            Ey(i)=Ey(i)+E(2,j,i)*solver.x(mesh.EdgeOfTet(numTet,j));
            Ez(i)=Ez(i)+E(3,j,i)*solver.x(mesh.EdgeOfTet(numTet,j));
            Hx(i)=Hx(i)+factor*curlE(1,j,i)*solver.x(mesh.EdgeOfTet(numTet,j));
            Hy(i)=Hy(i)+factor*curlE(2,j,i)*solver.x(mesh.EdgeOfTet(numTet,j));
            Hz(i)=Hz(i)+factor*curlE(3,j,i)*solver.x(mesh.EdgeOfTet(numTet,j));
        end
    end

    P=zeros(3,nbrGP);
    for i=1:nbrGP
        P(1,i) = 0.5 * (real(Ey(i))*real(Hz(i))+imag(Ey(i))*imag(Hz(i))-real(Ez(i))*real(Hy(i))-imag(Ez(i))*imag(Hy(i)));
        P(2,i) = 0.5 * (real(Ez(i))*real(Hx(i))+imag(Ez(i))*imag(Hx(i))-real(Ex(i))*real(Hz(i))-imag(Ex(i))*imag(Hz(i)));
        P(3,i) = 0.5 * (real(Ex(i))*real(Hy(i))+imag(Ex(i))*imag(Hy(i))-real(Ey(i))*real(Hx(i))-imag(Ey(i))*imag(Hx(i)));
        nP=nP+integCoe*weight(i)*P(3,i);
    end

    
end


end