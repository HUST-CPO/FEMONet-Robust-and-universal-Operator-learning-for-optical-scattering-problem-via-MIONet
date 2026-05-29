function [xx,yy,normHH,HHx,HHy,HHz]=get_mag2(triIndex,mesh,solver,k0)

nbr=length(triIndex);
xx=zeros(3,nbr);
yy=zeros(3,nbr);
normHH=zeros(3,nbr);
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
        xx(:,n)=[x(1);x(2);x(3)];
        yy(:,n)=[y(1);y(2);y(3)];
    elseif numFace==2 %124
        x2=[1;0;0];
        y2=[0;1;0];
        z2=[0;0;0];
        xx(:,n)=[x(1);x(2);x(4)];
        yy(:,n)=[y(1);y(2);y(4)];
    elseif numFace==3 %134
        x2=[1;0;0];
        y2=[0;0;0];
        z2=[0;1;0];
        xx(:,n)=[x(1);x(3);x(4)];
        yy(:,n)=[y(1);y(3);y(4)];
    elseif numFace==4 %234
        x2=[0;0;0];
        y2=[1;0;0];
        z2=[0;1;0];
        xx(:,n)=[x(2);x(3);x(4)];
        yy(:,n)=[y(2);y(3);y(4)];
    end

    %jac
    Jac=zeros(3,3);
    Jac(1,1)=x(1)-x(4);Jac(1,2)=y(1)-y(4);Jac(1,3)=z(1)-z(4);
    Jac(2,1)=x(2)-x(4);Jac(2,2)=y(2)-y(4);Jac(2,3)=z(2)-z(4);
    Jac(3,1)=x(3)-x(4);Jac(3,2)=y(3)-y(4);Jac(3,3)=z(3)-z(4);

    TJac = (Jac')/det(Jac);
    factor = 1i /(k0*120*pi);

    %bf
    curlE=zeros(3,6,3); % 3 x 6 basis functions x 3 nodes
    for i=1:3
        for j=1:6
            curlE(:,j,i)=getBF(2,j,x2(i),y2(i),z2(i));
            curlE(:,j,i)=TJac*curlE(:,j,i)*l(j);
        end
    end

    % Magnetic field
    Hx=zeros(3,1);Hy=zeros(3,1);Hz=zeros(3,1);
    for i=1:3
        for j=1:6
            Hx(i)=Hx(i)+factor*curlE(1,j,i)*solver.x(mesh.EdgeOfTet(numTet,j));
            Hy(i)=Hy(i)+factor*curlE(2,j,i)*solver.x(mesh.EdgeOfTet(numTet,j));
            Hz(i)=Hz(i)+factor*curlE(3,j,i)*solver.x(mesh.EdgeOfTet(numTet,j));
        end
    end
    normE=sqrt(abs(Hx.*conj(Hx)+Hy.*conj(Hy)+Hz.*conj(Hz)));
    normHH(:,n)=normE;
    HHx(:,n)=Hx;
    HHy(:,n)=Hy;
    HHz(:,n)=Hz;

end


end