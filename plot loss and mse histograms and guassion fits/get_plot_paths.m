function P = get_plot_paths()
%GET_PLOT_PATHS  Standard directories for loss / MSE figure scripts.
    P.script_dir = fileparts(mfilename('fullpath'));
    P.repo_root  = fileparts(P.script_dir);
    P.log_dir    = fullfile(P.script_dir, 'loss_log');
    P.out_dir    = fullfile(P.script_dir, 'plot_results');
    P.mse_dir    = fullfile(P.script_dir, 'mse_cache');
    P.ablation_dir = fullfile(P.script_dir, 'ablation');

    ensure_dir(P.out_dir);
    ensure_dir(P.mse_dir);
end

function mat_dir = case_mat_dir(P, case_folder)
%CASE_MAT_DIR  Path to a case study mat_data folder.
    mat_dir = fullfile(P.repo_root, case_folder, 'samples and post proceeding', 'mat_data');
end

function ensure_dir(d)
    if ~exist(d, 'dir')
        mkdir(d);
    end
end
