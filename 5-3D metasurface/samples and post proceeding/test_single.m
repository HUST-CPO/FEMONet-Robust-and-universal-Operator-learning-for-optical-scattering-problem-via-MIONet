clc
clear all
close all

mat_dir = fullfile(fileparts(mfilename('fullpath')), 'mat_data');

load(fullfile(mat_dir, 'Train_data_3Dcase3_261.mat'))
load(fullfile(mat_dir, 'deepOnet_data_3Dcase3_261.mat'))
load(fullfile(mat_dir, 'idx_3Dcase3_261.mat'))
load(fullfile(mat_dir, 'E_test_pred_size_261_ddp_fft.mat'))


i=4;
mesh = Matrix{7,testIdx(i)};
P =  Matrix{8,testIdx(i)};
solver.x=P*Ez_test(i,1:coord_len_test(i)).'/10;
solver.x=P*E_pred(i,1:coord_len_test(i)).'/10;
Et = solver.x;


figure('visible','on');

triIndex = findTri([12,20],mesh);
[x,y,normE,Ex,Ey,Ez] = get_ele2(triIndex,mesh,solver);

% x, y are typically 3 x Ntri; each column is one triangle
nTri = size(x,2);

% Merge duplicate nodes
XY = [x(:), y(:)];
[XYu,~,ic] = uniquetol(XY, 1e-10, 'ByRows', true);

Faces = reshape(ic, size(x)).';


% Average element values at shared physical nodes
Cnode = accumarray(ic, normE(:), [], @mean);

patch('Faces', Faces, ...
      'Vertices', XYu, ...
      'FaceVertexCData', Cnode, ...
      'FaceColor', 'interp', ...
      'EdgeColor', 'none');

axis equal tight
colormap jet
colorbar

