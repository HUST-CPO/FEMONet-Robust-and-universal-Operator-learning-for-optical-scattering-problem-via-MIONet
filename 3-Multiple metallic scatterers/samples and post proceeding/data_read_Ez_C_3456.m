clc; clear;

mat_dir = fullfile(fileparts(mfilename('fullpath')), 'mat_data');
script_dir = fileparts(mfilename('fullpath'));
if ~exist(mat_dir, 'dir'), mkdir(mat_dir); end

% Load merged C1+C2 raw data
load(fullfile(mat_dir, 'Train_data_C_3456.mat'))

% Max nnz length and max DOF length
max1 = 0;
max2 = 0;
nbr = size(Matrix, 2);
coord_len = zeros(nbr, 1);

for i = 1:nbr
    max1 = max(max1, size(Matrix{1, i}, 1));
    max2 = max(max2, size(Matrix{4, i}, 1));
    coord_len(i, 1) = size(Matrix{4, i}, 1);
end

Ai = zeros(nbr, max1);
Aj = zeros(nbr, max1);
Av = zeros(nbr, max1);
b_all = zeros(nbr, max2);
Ez = zeros(nbr, max2);
xx = zeros(nbr, max2, 2);

for i = 1:nbr
    Ai(i, 1:size(Matrix{1, i}, 1)) = Matrix{1, i}';
    Aj(i, 1:size(Matrix{2, i}, 1)) = Matrix{2, i}';
    Av(i, 1:size(Matrix{3, i}, 1)) = Matrix{3, i}.';
    b_all(i, 1:size(Matrix{4, i}, 1)) = Matrix{4, i}.';
    Ez(i, 1:size(Matrix{5, i}, 1)) = Matrix{5, i}.';
    xx(i, 1:size(Matrix{6, i}, 1), :) = Matrix{6, i};
end

% Reshape to network input (N,2,128,128)
Eplison = reshape(eps_all, nbr, 2, 128, 128);
Ebz = reshape(Ebz_all, nbr, 2, 128, 128);

% Reproducible split: 20% test, 80% train
rng(20260407);
pctTest = 0.2;
testSize = round(nbr * pctTest);
idx = randperm(nbr);
testIdx = idx(1:testSize);
trainIdx = idx(testSize+1:end);

% Split tensors
X_train = xx(trainIdx, :, :);
X_test  = xx(testIdx, :, :);

Eplison_train = Eplison(trainIdx, :, :, :);
Eplison_test  = Eplison(testIdx, :, :, :);

Ebz_train = Ebz(trainIdx, :, :, :);
Ebz_test  = Ebz(testIdx, :, :, :);

Ez_train = Ez(trainIdx, :);
Ez_test  = Ez(testIdx, :);

coord_len_train = coord_len(trainIdx, :);
coord_len_test  = coord_len(testIdx, :);

Ai_train = Ai(trainIdx, :);
Aj_train = Aj(trainIdx, :);
Av_train = Av(trainIdx, :);

Ai_test = Ai(testIdx, :);
Aj_test = Aj(testIdx, :);
Av_test = Av(testIdx, :);

b_train = b_all(trainIdx, :);
b_test  = b_all(testIdx, :);

% Split model_id labels (1:C1, 2:C2)
if exist('model_id', 'var')
    model_id_train = model_id(trainIdx, :);
    model_id_test  = model_id(testIdx, :);
else
    model_id = zeros(nbr, 1);
    model_id_train = model_id(trainIdx, :);
    model_id_test  = model_id(testIdx, :);
end

% Consistency check
i = 1;
mask = (Ai_train(i, :) > 0) & (Aj_train(i, :) > 0);
A = sparse(Ai_train(i, mask), Aj_train(i, mask), Av_train(i, mask));
r = A * Ez_train(i, 1:coord_len_train(i)).' - b_train(i, 1:coord_len_train(i)).';
r_norm = sum(abs(r));
fprintf('Check sample #1 residual sum(abs(Ax-b)) = %.6e\n', r_norm);

fprintf('Total samples: %d, train: %d, test: %d\n', nbr, length(trainIdx), length(testIdx));
fprintf('C1 in train: %d, C2 in train: %d\n', sum(model_id_train==1), sum(model_id_train==2));
fprintf('C1 in test : %d, C2 in test : %d\n', sum(model_id_test==1), sum(model_id_test==2));

% Save DeepONet input data
save(fullfile(mat_dir, 'deepOnet_data_C_3456.mat'), ...
    'X_train','X_test', ...
    'Eplison_train','Eplison_test','Ebz_train','Ebz_test', ...
    'Ez_train','Ez_test', ...
    'Ai_train','Ai_test','Aj_train','Aj_test','Av_train','Av_test', ...
    'b_train','b_test','coord_len_train','coord_len_test', ...
    'model_id_train','model_id_test', '-v7.3');

% Save indices for use with deepOnet_data_C_3456.mat
save(fullfile(mat_dir, 'idx_C_3456.mat'), 'trainIdx', 'testIdx');
