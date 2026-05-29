% Plot training loss curves for all cases (reads loss_log/, writes plot_results/).
clc; clear; close all;

P = get_plot_paths();

plot_loss_curve(fullfile(P.log_dir, 'modelA_loss.txt'),    'modelA_loss',    'epoch');
plot_loss_curve(fullfile(P.log_dir, 'modelB_loss.log'),    'modelB_loss',    'epoch');
plot_loss_curve(fullfile(P.log_dir, 'modelC_loss.txt'),    'modelC_loss',    'columns');
plot_loss_curve(fullfile(P.log_dir, 'model3D_loss.log'),   'model3D_loss',   'epoch');
plot_loss_curve(fullfile(P.log_dir, 'modelSPP1_loss.txt'), 'modelSPP1_loss', 'epoch');
plot_loss_curve(fullfile(P.log_dir, 'modelSPP14_loss.txt'),'modelSPP14_loss','epoch');

% Case A vs ablation (modelAV) comparison
plot_loss_modelAV();

fprintf('All loss curves exported to: %s\n', P.out_dir);
