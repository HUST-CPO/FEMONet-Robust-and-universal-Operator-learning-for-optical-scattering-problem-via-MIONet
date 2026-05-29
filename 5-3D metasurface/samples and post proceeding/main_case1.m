clc
clear all

mat_dir = fullfile(fileparts(mfilename('fullpath')), 'mat_data');
script_dir = fileparts(mfilename('fullpath'));
if ~exist(mat_dir, 'dir')
    mkdir(mat_dir);
end

%model
% Geometry parameters
%lda0=0.7;
lda0=0.54:0.001:0.8;

nbr1=length(lda0);
all_num=nbr1;
model=mphload(fullfile(script_dir,'comsol_project','doublePBC.mph'));

nn=0;
for l=1:nbr1
    nn=nn+1;

    % Update parameters
    model.param.set('lda0',string(lda0(l))+'[m]');

    % Update geometry
    geom=model.geom('geom1');
    geom.run();

    % Update mesh
    meshX=model.mesh('mesh1');
    meshX.run();

    % Run COMSOL study
    model.study('std1').run();

    n = 128;  % 128x128 grid
    x_range = linspace(-0.125, 0.125, n);
    y_range = linspace(0.125, -0.125, n);
    z_range = linspace(1, -1, n);
    [X, Y, Z] = ndgrid(x_range, y_range, z_range);
    points = [X(:), Y(:), Z(:)];

    reps=mphinterp(model, 'ewfd.n_iso', 'coord', points')';
    ieps=mphinterp(model, 'ewfd.ki_iso', 'coord', points')';
    reps = reshape(reps, n, n, n);
    ieps = reshape(ieps, n, n, n);
    eps = cat(4, reps, ieps);



    % rEbz=mphinterp(model, 'real(ewfd.Ebz)', 'coord', points')';
    % iEbz=mphinterp(model, 'imag(ewfd.Ebz)', 'coord', points')';
    % rEbz = reshape(rEbz, n, n, n);
    % iEbz = reshape(iEbz, n, n, n);
    % Ebz = cat(4, rEbz, iEbz);

    [m1, m2]=mphmeshstats(model);

    % Vertices
    mesh.NbrVertex=max(size(m2.vertex));% vertex count
    mesh.Vertex=m2.vertex'; % vertex coords 3*n

    % Tetrahedra
    mesh.NbrTet=max(size(m2.elem{2}));% tet count
    mesh.DomainOfTet=m2.elementity{2};% tet domain index n*1
    mesh.Tet=m2.elem{2}+1;% tet vertex indices 4*n
    mesh.Tet = sort(mesh.Tet)';

    % Edges
    % Edge ordering
    el2no=mesh.Tet';
    n1=el2no([1 1 1 2 2 3],:);
    n2=el2no([2 3 4 3 4 4],:);
    el_ed2no_array=[n1(:) n2(:)];
    % Order: 1-2 1-3 1-4 2-3 2-4 3-4
    % Edges global n*2; EdgeOfTet per-tet edge index 6N*1; sign edge direction 6N*1
    [mesh.Edge,~,mesh.EdgeOfTet]=unique(el_ed2no_array,'rows');
    mesh.NbrEdge=max(size(mesh.Edge));% edge count
    mesh.EdgeOfTet=reshape(mesh.EdgeOfTet,6,mesh.NbrTet)';

    % Faces
    el2no=mesh.Tet';
    n1=el2no([1 1 1 2],:);
    n2=el2no([2 2 3 3],:);
    n3=el2no([3 4 4 4],:);
    el_ed2no_array=[n1(:) n2(:) n3(:)];
    mesh.Tri=sort(m2.elem{3})'+1;
    mesh.DomainOfTri=m2.elementity{3};
    mesh.NbrTri=length(mesh.Tri);
    mesh.ConnOfTri=zeros(mesh.NbrTri,2);

    for i=1:mesh.NbrTri
        [is,index]=ismember(mesh.Tri(i,:),el_ed2no_array,"rows");
        mesh.ConnOfTri(i,1)=fix((index-1)/4)+1;
        mesh.ConnOfTri(i,2)=index-(mesh.ConnOfTri(i,1)-1)*4;
    end

    mesh.NormOfFace=zeros(27,3);
    mesh.NormOfFace(3,:)=[0,0,-1];
    mesh.NormOfFace(13,:)=[0,0,1];

    % Physical parameters
    phy.eps = zeros(1,5);
    phy.eps(1,2)=(mphinterp(model, 'ewfd.n_iso', 'coord', [0,0,-0.4]')-1i*mphinterp(model, 'ewfd.ki_iso', 'coord', [0,0,-0.4]'))^2;
    phy.eps(1,3)=(mphinterp(model, 'ewfd.n_iso', 'coord', [0,0,-0.2]')-1i*mphinterp(model, 'ewfd.ki_iso', 'coord', [0,0,-0.2]'))^2;
    phy.eps(1,5)=(mphinterp(model, 'ewfd.n_iso', 'coord', [0,0,0]')-1i*mphinterp(model, 'ewfd.ki_iso', 'coord', [0,0,0]'))^2;
    phy.eps(1,[1 4])=1;
    phy.mur=[1 1 1 1 1];
    phy.lda0=lda0(l);
    phy.out=[3];
    phy.inc=[13];
    phy.Einc=[1;0;0];
    mesh.incIndex=findTri(phy.inc,mesh);
    mesh.outIndex=findTri(phy.out,mesh);
    phy.src=[1 4 7 10];% boundary domain indices
    phy.dst=[24 25 26 27];
    mesh.PBCIndex=findPBCIndex(phy.src,phy.dst,[0.25;0;0],mesh);
    mesh.PBCphi=1;
    phy.src2=[2 5 8 11];% boundary domain indices
    phy.dst2=[14 15 16 17];
    mesh.PBCIndex2=findPBCIndex(phy.src2,phy.dst2,[0;0.25;0],mesh);
    mesh.PBCphi2=1;

    % Matrix assembly
    solver.Ai=[];solver.Aj=[];solver.Av=[];
    solver.b=zeros(mesh.NbrEdge,1);
    solver=assembly_equ(phy,mesh,solver);
    % Outgoing boundary
    solver=assembly_out(phy,mesh,solver);
    % Incident boundary
    solver=assembly_inc(phy,mesh,solver);
    solver.A=sparse(solver.Ai,solver.Aj,solver.Av);
    %Bloch boundary
    [mesh,solver]=assembly_pbc_double(mesh,solver);

    % Solve linear system
    solver.A=solver.P'*solver.A*solver.P;
    % Scale RHS
    solver.b=-solver.P'*solver.b*10;
    solver.x0=solver.A\solver.b;
    solver.x=solver.P*solver.x0;

    % Electric field
    % triIndex=findTri(13,mesh);
    % [x,y,normE,Ex,Ey,Ez]=get_ele2(triIndex,mesh,solver);
    % patch(x,y,normE,'EdgeAlpha',0)
    % colormap jet;
    % colorbar
    % min(min(normE))
    % max(max(normE))

    edge_temp=[mesh.Vertex(mesh.Edge(mesh.Index,1),:),mesh.Vertex(mesh.Edge(mesh.Index,2),:)];
    edge_temp=[edge_temp(:,1)/0.125,edge_temp(:,2)/0.125,edge_temp(:,3),...
        edge_temp(:,4)/0.125,edge_temp(:,5)/0.125,edge_temp(:,6)];



    [Ai,Aj,Av]=find(solver.A);
    Matrix{1,nn}= Ai;
    Matrix{2,nn}= Aj;
    Matrix{3,nn}= Av;
    Matrix{4,nn}= solver.b;
    Matrix{5,nn}= solver.x0;
    Matrix{6,nn}= edge_temp;
    Matrix{7,nn}= mesh;
    Matrix{8,nn}= solver.P;
    lda_all(nn,1)=lda0(l);
    eps_all(nn,:,:,:,:)=eps;
    %Ebz_all(nn,:,:,:,:)=Ebz;
    close all;
end


save(fullfile(mat_dir, "Train_data_3Dcase3_521"),"Matrix","eps_all","lda_all",'phy',"-v7.3");