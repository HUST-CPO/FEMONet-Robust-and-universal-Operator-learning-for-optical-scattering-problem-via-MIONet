function compute_mse_cache(case_id)
%COMPUTE_MSE_CACHE  Per-sample MSE vectors for histogram / Gaussian fits.
%   Writes mse_cache/<case_id>_test.mat and _train.mat
%
%   case_id: 'A' | '3D' | 'C' | 'B' | 'SPP14' | 'SPP1'

    P = get_plot_paths();
    cfg = case_config(case_id, P);

    load(fullfile(cfg.mat_dir, cfg.train_data), 'Matrix');
    Sd = load(fullfile(cfg.mat_dir, cfg.deepOnet));
    load(fullfile(cfg.mat_dir, cfg.idx), 'testIdx', 'trainIdx');

    coord_len_test = Sd.coord_len_test;
    coord_len_train = Sd.coord_len_train;
    testIds = testIdx(:);
    trainIds = trainIdx(:);

    St = load(fullfile(cfg.mat_dir, cfg.E_test));
    Su = load(fullfile(cfg.mat_dir, cfg.E_train));
    E_pred_test = St.E_pred;
    E_pred_train = Su.E_pred;

  % --- test set ---
    num_test = size(E_pred_test, 1);
    sample_mse = zeros(num_test, 1);
    for i = 1:num_test
        L = coord_len_test(i);
        E_true = Matrix{cfg.matrix_row, testIds(i)} * cfg.field_scale;
        E_nn = E_pred_test(i, 1:L).' * cfg.field_scale;
        sample_mse(i) = mean(abs(E_true - E_nn).^2);
    end
    mse_global = sum(sample_mse .* coord_len_test(:)) / sum(coord_len_test);
    out_test = fullfile(P.mse_dir, [case_id, '_test.mat']);
    save(out_test, 'sample_mse', 'mse_global');
    fprintf('%s test cache: %s (global MSE = %.6g)\n', case_id, out_test, mse_global);

  % --- train set ---
    num_train = size(E_pred_train, 1);
    sample_mse = zeros(num_train, 1);
    for i = 1:num_train
        L = coord_len_train(i);
        E_true = Matrix{cfg.matrix_row, trainIds(i)} * cfg.field_scale;
        E_nn = E_pred_train(i, 1:L).' * cfg.field_scale;
        sample_mse(i) = mean(abs(E_true - E_nn).^2);
    end
    mse_global = sum(sample_mse .* coord_len_train(:)) / sum(coord_len_train);
    out_train = fullfile(P.mse_dir, [case_id, '_train.mat']);
    save(out_train, 'sample_mse', 'mse_global');
    fprintf('%s train cache: %s (global MSE = %.6g)\n', case_id, out_train, mse_global);
end

function cfg = case_config(case_id, P)
    switch upper(case_id)
        case 'A'
            cfg.case_folder = '1-Basic lossless scatterers';
            cfg.train_data = 'Train_data_A_1558.mat';
            cfg.deepOnet = 'deepOnet_data_A_1558.mat';
            cfg.idx = 'idx_A_1558.mat';
            cfg.E_test = 'E_test_pred_size_1558.mat';
            cfg.E_train = 'E_train_pred_size_1558.mat';
            cfg.matrix_row = 5;
            cfg.field_scale = 1;
        case '3D'
            cfg.case_folder = '5-3D metasurface';
            cfg.train_data = 'Train_data_3Dcase3_261.mat';
            cfg.deepOnet = 'deepOnet_data_3Dcase3_261.mat';
            cfg.idx = 'idx_3Dcase3_261.mat';
            cfg.E_test = 'E_test_pred_size_261_ddp_fft.mat';
            cfg.E_train = 'E_train_pred_size_261_ddp_fft.mat';
            cfg.matrix_row = 5;
            cfg.field_scale = 1/10;
        case 'C'
            cfg.case_folder = '3-Multiple metallic scatterers';
            cfg.train_data = 'Train_data_C_3456.mat';
            cfg.deepOnet = 'deepOnet_data_C_3456.mat';
            cfg.idx = 'idx_C_3456.mat';
            cfg.E_test = 'E_test_pred_size_3456.mat';
            cfg.E_train = 'E_train_pred_size_3456.mat';
            cfg.matrix_row = 5;
            cfg.field_scale = 1;
        case 'B'
            cfg.case_folder = '2-Single metallic scatterers';
            cfg.train_data = 'Train_data_B1_50688.mat';
            cfg.deepOnet = 'deepOnet_data_B1_50688.mat';
            cfg.idx = 'idx_B1_50688.mat';
            cfg.E_test = 'E_test_pred_size_50688_ddp.mat';
            cfg.E_train = 'E_train_pred_size_50688_ddp.mat';
            cfg.matrix_row = 5;
            cfg.field_scale = 1;
        case 'SPP14'
            cfg.case_folder = '4-SPP in plasmonic nanostructure';
            cfg.train_data = 'Train_data_SPP14_41.mat';
            cfg.deepOnet = 'deepOnet_data_SPP14_41.mat';
            cfg.idx = 'idx_SPP14_41.mat';
            cfg.E_test = 'E_test_pred_size_41.mat';
            cfg.E_train = 'E_train_pred_size_41.mat';
            cfg.matrix_row = 5;
            cfg.field_scale = 1;
        case 'SPP1'
            cfg.case_folder = 's1-SPP in plasmonic nanostructure';
            cfg.train_data = 'Train_data_SPP1_71.mat';
            cfg.deepOnet = 'deepOnet_data_SPP1_71.mat';
            cfg.idx = 'idx_SPP1_71.mat';
            cfg.E_test = 'E_test_pred_size_71.mat';
            cfg.E_train = 'E_train_pred_size_71.mat';
            cfg.matrix_row = 5;
            cfg.field_scale = 1;
        otherwise
            error('Unknown case_id: %s', case_id);
    end
    cfg.mat_dir = case_mat_dir(P, cfg.case_folder);
    required = {cfg.train_data, cfg.deepOnet, cfg.idx, cfg.E_test, cfg.E_train};
    for k = 1:numel(required)
        fpath = fullfile(cfg.mat_dir, required{k});
        if ~isfile(fpath)
            error('Missing input file for case %s:\n  %s', case_id, fpath);
        end
    end
end
