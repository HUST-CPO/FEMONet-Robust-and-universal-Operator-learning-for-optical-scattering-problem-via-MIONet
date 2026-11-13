% Reuse the exact paper postprocessor; no new solve or interpolation of coefficients.
root = fileparts(mfilename('fullpath'));
project = fileparts(root);
src = fullfile(project,'SPP14');
addpath(fullfile(src,'post'), fullfile(src,'kernel'), fullfile(src,'function'));
d = load(fullfile(src,'deepOnet_data_SPP14_41.mat'));
ids = load(fullfile(src,'idx_SPP14_41.mat'));
meshdata = load(fullfile(src,'Train_data_SPP14_41.mat'),'Matrix');
out = fullfile(root,'field_comparison');
if ~exist(out,'dir'), mkdir(out); end
wavelengths = [535 655 530 650];
Fields = struct([]);
for k = 1:numel(wavelengths)
    wavelength = wavelengths(k);
    split = 'test';
    i = find(abs(d.Lambda_test*500-wavelength)<1e-8);
    if isempty(i)
        split = 'train';
        i = find(abs(d.Lambda_train*500-wavelength)<1e-8);
    end
    assert(numel(i)==1);
    sid = ids.([split 'Idx'])(i);
    Mesh = meshdata.Matrix{7,sid};
    n = d.(['coord_len_' split])(i);
    truth = d.(['Ez_' split])(i,1:n).';
    assert(max(abs(truth-meshdata.Matrix{5,sid}(:)))<1e-12);
    archived = load(fullfile(src,['E_' split '_pred_size_41.mat']),'E_pred');
    ilu = load(fullfile(root,'runs','ilu_seed123',['E_' split '_pred.mat']),'E_pred');
    raw = load(fullfile(root,'runs','raw_seed123',['E_' split '_pred.mat']),'E_pred');
    coefficients = [truth archived.E_pred(i,1:n).' ilu.E_pred(i,1:n).' raw.E_pred(i,1:n).'];
    Ex = zeros(size(Mesh.vertex,1),4); Ey = Ex;
    for j = 1:4
        [Ex(:,j),Ey(:,j)] = GetExEy(Mesh.vertex,Mesh.tri,Mesh.edgesOfTri,coefficients(:,j));
    end
    Fields(k).wavelength_nm = wavelength;
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
disp('Reconstructed four wavelengths using original GetExEy.m');
