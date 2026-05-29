clc
clear all

%model
% Geometry parameters
%factor=0.8;
factor=0.2:0.01:0.9;

nbr1=length(factor);

mat_dir = fullfile(fileparts(mfilename('fullpath')), 'mat_data');
script_dir = fileparts(mfilename('fullpath'));
if ~exist(mat_dir, 'dir')
    mkdir(mat_dir);
end

addpath(fullfile(script_dir, 'function'));
addpath(fullfile(script_dir, 'kernel'));
addpath(fullfile(script_dir, 'mesh'));
addpath(fullfile(script_dir, 'post'));
fileName="case1_2D";
model=mphload(fullfile(script_dir, 'comsol_project', fileName + ".mph"));


nn=0;
for l=1:nbr1
    nn=nn+1;

    % Update parameters
    model.param.set('factor',string(factor(l)));

    % Update geometry
    geom=model.geom('geom1');
    geom.run();

    % Update mesh
    mesh=model.mesh('mesh1');
    mesh.run();

    % Solve
    model.study('std1').run();


    % Mesh info
    meshInfo = mphxmeshinfo(model);
    vertex=meshInfo.dofs.coords';
    tri=sort(meshInfo.elements.tri.dofs'+1,2);
    n = 128;  % 128x128 grid
    x_range = linspace(0, 0.94262, n);
    y_range = linspace(1.4857, 0, n);
    [X, Y] = meshgrid(x_range, y_range);
    points = [X(:), Y(:)];
    reps=mphinterp(model, 'ewfd.n_iso', 'coord', points')';
    ieps=mphinterp(model, 'ewfd.ki_iso', 'coord', points')';
    reps = reps/max(abs(reps));
    ieps = ieps/max(abs(ieps));
    eps=[reps ieps];
    eps = reshape(eps, 128, 128, 2);
%% 

    [~, m2]=mphmeshstats(model);
    %process mesh data
    Mesh.vertex=m2.vertex';%coordinate of nodes  unit:m
    tri=m2.elem{2}+1;
    Mesh.tri = sort(tri)';%tri
    Mesh.triID=m2.elementity{2};%face ID of tri
    Mesh.Bedge=m2.elem{1}'+1;%boundary edge
    Mesh.BedgeID=m2.elementity{1};%edge Id of boundary edge
    Mesh.nbrVertex=length(Mesh.vertex);
    Mesh.nbrTri=length(Mesh.tri);
    %treat mesh data
    Mesh=GetEdge(Mesh);
    Mesh.Dof=Mesh.nbrEdges;
    %boundary condition setting
    Mesh.inc=FindEdge([7],Mesh);
    Mesh.out=FindEdge([2],Mesh);

    %physic model
    physic.c_const=299792458;
    physic.lam0=1;%wavelength
    physic.k0=2*pi/physic.lam0;
    %material
    physic.epsilonr=ones(5,1);%Corresponding to triID
    physic.epsilonr([2,5])=(0.26841-1i*6.011)^2;
    physic.epsilonr([1,4])=(1.55)^2;
    physic.mur=ones(5,1);%Corresponding to triID

    targetflag1=[1,3,5];
    targetflag2=[18,19,20];
    find1=findTris(targetflag1,Mesh);
    find2=findTris(targetflag2,Mesh);
    [copyOfBedge,nodephi]=GetcopyOfBedge(find1,find2,Mesh);
    DstNodeIndex = copyOfBedge(:,2);
    SrcNodeIndex = copyOfBedge(:,1);
    DeleteIndex=unique(sort(DstNodeIndex));
    Index=1:(Mesh.Dof);
    Index(DeleteIndex)=[];
    P=speye(Mesh.Dof);
    for i = 1:size(DstNodeIndex,1)
        P(DstNodeIndex(i,1),SrcNodeIndex(i,1))=nodephi(i);
    end
    P=P(:,Index);



    %init matrix
    solver.Ai=[];
    solver.Aj=[];
    solver.Av=[];
    solver.b=complex(zeros(Mesh.Dof,1));

    %assemble
    solver=AssemblyOfEqu(Mesh,physic,solver);
    solver=AssemblyOfInc(Mesh,physic,solver);
    solver=AssemblyOfOut(Mesh,physic,solver);

    %solution
    solver.A=sparse(solver.Ai,solver.Aj,solver.Av);
    solver.A=P'*solver.A*P;
    solver.b=P'*solver.b;
    solver.x0=solver.A\solver.b;
    solver.x=P*solver.x0;

    %post
    Et=solver.x;
    [Ex,Ey]=GetExEy(Mesh.vertex,Mesh.tri,Mesh.edgesOfTri,Et);


    %plot
    PlotnormE(Mesh.vertex,Mesh.tri,Ex,Ey,0,0);

    % edge_temp=Mesh.vertex(Mesh.edges(Index,1),:)/2+Mesh.vertex(Mesh.edges(Index,2),:)/2;
    % edge_temp=[(2*edge_temp(:,1)-0.94262)/0.94262,(2*edge_temp(:,2)-1.4857)/1.4857];
    edge_temp=[Mesh.vertex(Mesh.edges(Index,1),:),Mesh.vertex(Mesh.edges(Index,2),:)];
    edge_temp=[(2*edge_temp(:,1)-0.94262)/0.94262,(2*edge_temp(:,2)-1.4857)/1.4857,...
        (2*edge_temp(:,3)-0.94262)/0.94262,(2*edge_temp(:,4)-1.4857)/1.4857];



    [Ai,Aj,Av]=find(solver.A);
    Matrix{1,nn}= Ai;
    Matrix{2,nn}= Aj;
    Matrix{3,nn}= Av;
    Matrix{4,nn}= solver.b;
    Matrix{5,nn}= solver.x0;
    Matrix{6,nn}= edge_temp;
    Matrix{7,nn}= Mesh;
    Matrix{8,nn}= P;
    eps_all(nn,:,:,:)=eps;
    close all;
end

save(fullfile(mat_dir, "Train_data_SPP1_71"),"Matrix","eps_all");