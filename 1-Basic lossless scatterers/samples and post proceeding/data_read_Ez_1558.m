clc; clear

mat_dir = fullfile(fileparts(mfilename('fullpath')), 'mat_data');
if ~exist(mat_dir, 'dir')
    mkdir(mat_dir);
end

load(fullfile(mat_dir, 'Train_data_A_1558.mat'))

max1=0;
max2=0;
nbr = size(Matrix,2);
for i = 1:nbr
    if (max1<size(Matrix{1,i},1))
        max1=size(Matrix{1,i},1);
    end
    if (max2<size(Matrix{4,i},1))
        max2=size(Matrix{4,i},1);
    end
    coord_len(i,1)=size(Matrix{4,i},1);
end

Ai = zeros(nbr,max1);
Aj = zeros(nbr,max1);
Av = zeros(nbr,max1);
b_all = zeros(nbr,max2);
Ez = zeros(nbr,max2);
xx = zeros(nbr,max2,2);

for i = 1:nbr
    Ai(i,1:size(Matrix{1,i},1))=Matrix{1,i}';
    Aj(i,1:size(Matrix{2,i},1))=Matrix{2,i}';
    Av(i,1:size(Matrix{3,i},1))=Matrix{3,i}.';
    b_all(i,1:size(Matrix{4,i},1)) = Matrix{4,i}.';
    Ez(i,1:size(Matrix{5,i},1)) = Matrix{5,i}.';
    xx(i,1:size(Matrix{6,i},1),:) = Matrix{6,i};
end

Eplison = reshape(eps_all(:,:,:,1),nbr,1,128,128);

load(fullfile(mat_dir, 'idx_A_1558.mat'))

X_train = xx(trainIdx,:,:);
X_test  = xx(testIdx,:,:);

Eplison_train = Eplison(trainIdx, :, :,:);
Eplison_test = Eplison(testIdx, :, :,:);

Ez_train = Ez(trainIdx, :);
Ez_test = Ez(testIdx, :);

coord_len_train = coord_len(trainIdx, :);
coord_len_test = coord_len(testIdx, :);

Ai_train = Ai(trainIdx,:);
Aj_train = Aj(trainIdx,:);
Av_train = Av(trainIdx,:);

Ai_test = Ai(testIdx,:);
Aj_test = Aj(testIdx,:);
Av_test = Av(testIdx,:);

b_train = b_all(trainIdx,:);
b_test = b_all(testIdx,:);

save(fullfile(mat_dir, 'deepOnet_data_A_1558.mat'), ...
     'X_train','X_test', ...
     'Eplison_train','Eplison_test', ...
     'Ez_train','Ez_test', ...
     'Ai_train','Ai_test', 'Aj_train','Aj_test','Av_train','Av_test', ...
     'b_train','b_test','coord_len_train','coord_len_test');
