% Reuse the exact paper postprocessor; no new solve or interpolation of coefficients.
root = fileparts(mfilename('fullpath'));
project = fileparts(root);
src = fullfile(project,'SPP1');
addpath(fullfile(src,'post'), fullfile(src,'kernel'), fullfile(src,'function'));
d = load(fullfile(src,'deepOnet_data_SPP1_71.mat'));
ids = load(fullfile(src,'idx_SPP1_71.mat'));
meshdata = load(fullfile(src,'Train_data_SPP1_71.mat'),'Matrix');
out = fullfile(root,'field_comparison');
if ~exist(out,'dir'), mkdir(out); end
duty_cycles = [0.60 0.83];
Fields = struct([]);
for k = 1:numel(duty_cycles)
    duty_cycle = duty_cycles(k);
    split = 'test';
    i = find(abs((0.2+0.01*(double(ids.testIdx)-1))-duty_cycle)<1e-8);
    if isempty(i)
        split = 'train';
        i = find(abs((0.2+0.01*(double(ids.trainIdx)-1))-duty_cycle)<1e-8);
    end
    assert(numel(i)==1);
    sid = ids.([split 'Idx'])(i);
    Mesh = meshdata.Matrix{7,sid};
    n = d.(['coord_len_' split])(i);
    truth = d.(['Ez_' split])(i,1:n).';
    assert(max(abs(truth-meshdata.Matrix{5,sid}(:)))<1e-12);
    archived = load(fullfile(src,['E_' split '_pred_size_71.mat']),'E_pred');
    ilu = load(fullfile(root,'runs','ilu_seed123',['E_' split '_pred.mat']),'E_pred');
    raw = load(fullfile(root,'runs','raw_seed123',['E_' split '_pred.mat']),'E_pred');
    coefficients = [truth archived.E_pred(i,1:n).' ilu.E_pred(i,1:n).' raw.E_pred(i,1:n).'];
    P = meshdata.Matrix{8,sid};
    full_coefficients = P*coefficients;
    Ex = zeros(size(Mesh.vertex,1),4); Ey = Ex;
    for j = 1:4
        [Ex(:,j),Ey(:,j)] = GetExEy(Mesh.vertex,Mesh.tri,Mesh.edgesOfTri,full_coefficients(:,j));
    end
    Fields(k).duty_cycle = duty_cycle;
    Fields(k).split = split;
    Fields(k).split_index = i;
    Fields(k).sample_id = sid;
    Fields(k).vertex = Mesh.vertex;
    Fields(k).tri = Mesh.tri;
    Fields(k).Ex = Ex; Fields(k).Ey = Ey;
    Fields(k).coefficients = coefficients;
    Fields(k).normE = sqrt(abs(Ex).^2+abs(Ey).^2);
end
save(fullfile(out,'reconstructed_fields.mat'),'Fields','-v7');
disp('Reconstructed S2 duty cycles, with periodic prolongation P, using original GetExEy.m');
