clc
clear all

mat_dir = fullfile(fileparts(mfilename('fullpath')), 'mat_data');
script_dir = fileparts(mfilename('fullpath'));
if ~exist(mat_dir, 'dir')
    mkdir(mat_dir);
end

model=mphload(fullfile(script_dir,'comsol_project','doublePBC.mph'));
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

mesh.NormOfFace=zeros(58,3);
mesh.NormOfFace(3,:)=[0,0,-1];
mesh.NormOfFace(16,:)=[0,0,1];

save(fullfile(mat_dir, 'doublePBC_mesh.mat'),"mesh");