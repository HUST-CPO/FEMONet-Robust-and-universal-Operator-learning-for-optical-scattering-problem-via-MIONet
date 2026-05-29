function solver=AssemblyOfOut4(mesh,physic,solver)

k0=physic.k0;

%guass points
[lx,lp,IntegralEdgeOrder]=GuassPoints_Line(4);

%init
NbrSBC=length(mesh.out);
NbrNonA=NbrSBC;
NumNonA=0;
Ai=zeros(NbrNonA,1);
Aj=zeros(NbrNonA,1);
Av=complex(zeros(NbrNonA,1));

%loop of tri
for n=1:NbrSBC
    %edge
    numEle=mesh.out(n,1);
    numEdg=mesh.out(n,2);
    %coordinate
    xx=zeros(5,1);yy=zeros(5,1);
    xx(1:3)=mesh.vertex(mesh.tri(numEle,:),1);
    yy(1:3)=mesh.vertex(mesh.tri(numEle,:),2);
    
    if numEdg==1
        xx(4)=xx(1);xx(5)=xx(2);
        yy(4)=yy(1);yy(5)=yy(2);
        l=sqrt((xx(1)-xx(2))*(xx(1)-xx(2))+(yy(1)-yy(2))*(yy(1)-yy(2)));
    elseif numEdg==2
        xx(4)=xx(1);xx(5)=xx(3);
        yy(4)=yy(1);yy(5)=yy(3);
        l=sqrt((xx(1)-xx(3))*(xx(1)-xx(3))+(yy(1)-yy(3))*(yy(1)-yy(3)));
    else
        xx(4)=xx(2);xx(5)=xx(3);
        yy(4)=yy(2);yy(5)=yy(3);
        l=sqrt((xx(3)-xx(2))*(xx(3)-xx(2))+(yy(3)-yy(2))*(yy(3)-yy(2)));
    end

    %jac
    Jac=zeros(3,3);
    Jac(1,1)=xx(2)-xx(1);Jac(1,2)=yy(2)-yy(1);
    Jac(2,1)=xx(3)-xx(1);Jac(2,2)=yy(3)-yy(1);Jac(3,3)=1;
    InvJac=inv(Jac);

    phi1=(yy(5)-yy(4));phi2=(xx(4)-xx(5));
    if xx(5)~=xx(4)
        PhysicsFactor=(phi1)/(-phi2);
        PhysicsFactor=abs(sqrt(1+PhysicsFactor*PhysicsFactor)*(-phi2)/2);
    elseif xx(5)==xx(4)
        PhysicsFactor=abs(phi1)/2;
    end

    %normal
    normal=[1,0,0];

    u=zeros(IntegralEdgeOrder,1);v=zeros(IntegralEdgeOrder,1);
    if numEdg==1
        for i=1:IntegralEdgeOrder
            u(i)=(lx(i)+1)/2;
            v(i)=0;
        end
    elseif numEdg==2
        for i=1:IntegralEdgeOrder
            u(i)=0;
            v(i)=(lx(i)+1)/2;
        end
    else
        for i=1:IntegralEdgeOrder
            u(i)=(lx(i)+1)/2;
            v(i)=(1-lx(i))/2;
        end
    end

    %mapping
    mappingIndex=mesh.edgesOfTri(numEle,numEdg);

    %bf
    Et=zeros(3,IntegralEdgeOrder);
    for k=1:IntegralEdgeOrder
        temp=zeros(3,1);
        [temp(1),temp(2)]=BF_Et(numEdg,u(k),v(k));
        Et(:,k)=InvJac*temp*l;
    end

    %material
    domain=mesh.triID(numEle);
    eps=physic.epsilonr(domain);
    refra=sqrt(eps);

    %submatrix
    PhySBC=1i*k0*refra;
    Ae=complex(0);
    for k=1:IntegralEdgeOrder
        Ae=Ae+lp(k)*PhysicsFactor*PhySBC*dot(cross(normal,Et(:,k)),cross(normal,Et(:,k)));
    end

    %assemble
    NumNonA=NumNonA+1;
    Ai(NumNonA)=mappingIndex;
    Aj(NumNonA)=mappingIndex;
    Av(NumNonA)=Ae;
end

solver.Ai=[solver.Ai;Ai];
solver.Aj=[solver.Aj;Aj];
solver.Av=[solver.Av;Av];

end