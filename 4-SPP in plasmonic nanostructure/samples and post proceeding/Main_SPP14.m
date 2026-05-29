clc
clear all

%model
% Geometry parameters
%factor=0.8;
lda1=1:0.01:1.4;
%lda1=1;

nbr1=length(lda1);
all_num=nbr1;

mat_dir = fullfile(fileparts(mfilename('fullpath')), 'mat_data');
script_dir = fileparts(mfilename('fullpath'));
if ~exist(mat_dir, 'dir')
    mkdir(mat_dir);
end

addpath(fullfile(script_dir, 'function'));
addpath(fullfile(script_dir, 'kernel'));
addpath(fullfile(script_dir, 'mesh'));
addpath(fullfile(script_dir, 'post'));
fileName="case14_2D_freq";
model=mphload(fullfile(script_dir, 'comsol_project', fileName + ".mph"));

tic;
nn=0;
for l=1:nbr1
    nn=nn+1;

    % Update parameters
    model.param.set('lda1',string(lda1(l)));

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
    x_range = linspace(-0.124, 0.124, n);
    y_range = linspace(0.16, -0.18, n);
    [X, Y] = meshgrid(x_range, y_range);
    points = [X(:), Y(:)];
    reps=mphinterp(model, 'ewfd.n_iso', 'coord', points')';
    ieps=mphinterp(model, 'ewfd.ki_iso', 'coord', points')';
    % reps = reps/max(abs(reps));
    % ieps = ieps/max(abs(ieps));
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

    %physic model
    physic.c_const=299792458;
    physic.lam0=1;%wavelength
    physic.k0=2*pi/physic.lam0;
    %material
    physic.epsilonr=ones(5,1);%Corresponding to triID
    physic.epsilonr([1,5])=(mphinterp(model, 'ewfd.n_iso', 'coord', [0;-0.1])-1i*mphinterp(model, 'ewfd.ki_iso', 'coord', [0;-0.1]))^2;
    physic.epsilonr([2,4])=(1.5)^2;
    physic.mur=ones(5,1);%Corresponding to triID

    %init matrix
    solver.Ai=[];
    solver.Aj=[];
    solver.Av=[];
    solver.b=complex(zeros(Mesh.Dof,1));

    %assemble
    solver=AssemblyOfEqu(Mesh,physic,solver);
    Mesh.inc=FindEdge([7],Mesh);
    solver=AssemblyOfInc(Mesh,physic,solver);
    Mesh.out=FindEdge([2],Mesh);
    solver=AssemblyOfOut(Mesh,physic,solver);
    Mesh.out=FindEdge([1,3,5],Mesh);
    solver=AssemblyOfOut3(Mesh,physic,solver);
    Mesh.out=FindEdge([20,21,22],Mesh);
    solver=AssemblyOfOut4(Mesh,physic,solver);

    %solution
    solver.A=sparse(solver.Ai,solver.Aj,solver.Av);
    solver.x=solver.A\solver.b;

    %post
    Et=solver.x;
    [Ex,Ey]=GetExEy(Mesh.vertex,Mesh.tri,Mesh.edgesOfTri,Et);


    %plot
    PlotnormE(Mesh.vertex,Mesh.tri,Ex,Ey,0,0);

    edge_temp=[Mesh.vertex(Mesh.edges(:,1),:),Mesh.vertex(Mesh.edges(:,2),:)];
    edge_temp=[(edge_temp(:,1))/0.124,(edge_temp(:,2)+0.01)/0.17,...
        edge_temp(:,3)/0.124,(edge_temp(:,4)+0.01)/0.17];



    [Ai,Aj,Av]=find(solver.A);
    Matrix{1,nn}= Ai;
    Matrix{2,nn}= Aj;
    Matrix{3,nn}= Av;
    Matrix{4,nn}= solver.b;
    Matrix{5,nn}= solver.x;
    Matrix{6,nn}= edge_temp;
    Matrix{7,nn}= Mesh;
    eps_all(nn,:,:,:)=eps;
    lda_all(nn,1)=lda1(l);
    close all;
end
run_time = toc;

save(fullfile(mat_dir, "Train_data_SPP14_41"),"Matrix","eps_all","lda_all");