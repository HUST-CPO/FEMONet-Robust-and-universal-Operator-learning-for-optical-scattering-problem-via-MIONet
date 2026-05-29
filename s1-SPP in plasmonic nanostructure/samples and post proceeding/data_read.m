clc; clear

mat_dir = fullfile(fileparts(mfilename('fullpath')), 'mat_data');
script_dir = fileparts(mfilename('fullpath'));
if ~exist(mat_dir, 'dir')
    mkdir(mat_dir);
end

addpath(fullfile(script_dir, 'function'));
addpath(fullfile(script_dir, 'kernel'));
addpath(fullfile(script_dir, 'mesh'));
addpath(fullfile(script_dir, 'post'));

load(fullfile(mat_dir, 'Train_data_SPP1_71.mat'))

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
xx = zeros(nbr,max2,4);

for i = 1:nbr
    Ai(i,1:size(Matrix{1,i},1))=Matrix{1,i}';
    Aj(i,1:size(Matrix{2,i},1))=Matrix{2,i}';
    Av(i,1:size(Matrix{3,i},1))=Matrix{3,i}.';
    b_all(i,1:size(Matrix{4,i},1)) = Matrix{4,i}.';
    Ez(i,1:size(Matrix{5,i},1)) = Matrix{5,i}.';
    xx(i,1:size(Matrix{6,i},1),:) = Matrix{6,i};
end

Eplison = reshape(eps_all,nbr,2,128,128);

% Random train/test split
pctTrain = 0.2;
testSize = round(nbr * pctTrain);
idx = randperm(nbr);
% Train/test indices
testIdx = idx(1:testSize);  % First fraction is test set
%testIdx = idx(testSize+1:end);  % Remainder is train
trainIdx = idx(testSize+1:end);  % Remainder is train set

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

i = 1;
mask = (Ai_train(i,:) > 0) & (Aj_train(i,:) > 0);
A = sparse(Ai_train(i,mask),Aj_train(i,mask),Av_train(i,mask));
r = A*Ez_train(i,1:coord_len_train(i)).'- b_train(i,1:coord_len_train(i)).';
r_norm = (sum(abs(r)));
i=1;

Mesh=Matrix{7,trainIdx(i)};
P=Matrix{8,trainIdx(i)};
Et = P*Ez_train(i,1:coord_len_train(i)).';
[Ex,Ey]=GetExEy(Mesh.vertex,Mesh.tri,Mesh.edgesOfTri,Et);
PlotnormE(Mesh.vertex,Mesh.tri,Ex,Ey,0,0);


save(fullfile(mat_dir, 'deepOnet_data_SPP1_71.mat'), ...
     'X_train','X_test', ...
     'Eplison_train','Eplison_test', ...
     'Ez_train','Ez_test', ...
     'Ai_train','Ai_test', 'Aj_train','Aj_test','Av_train','Av_test', ...
     'b_train','b_test','coord_len_train','coord_len_test');      
save(fullfile(mat_dir, 'idx_SPP1_71.mat'),'trainIdx','testIdx');