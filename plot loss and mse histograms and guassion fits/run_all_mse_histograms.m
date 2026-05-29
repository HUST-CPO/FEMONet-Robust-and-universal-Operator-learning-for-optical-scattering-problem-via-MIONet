% Build MSE caches and Gaussian-fit histograms for cases with complete mat_data.
clc; clear; close all;

P = get_plot_paths();

% Cases with data available in the repository (May 2026 snapshot)
cases_ready = {'A', 'SPP14', 'SPP1'};
cases_optional = {'3D', 'C', 'B'};  % need Train_data / E_pred not in repo

for k = 1:numel(cases_ready)
    id = cases_ready{k};
    try
        compute_mse_cache(id);
        plot_mse_gaussian(id, [-6 0], ['error_MSE_', id]);
    catch ME
        warning('Case %s skipped: %s', id, ME.message);
    end
end

fprintf('Finished. Figures in: %s\n', P.out_dir);
fprintf('Optional cases (%s) require missing .mat files — see README.md.\n', ...
    strjoin(cases_optional, ', '));
