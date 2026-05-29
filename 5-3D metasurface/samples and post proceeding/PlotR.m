clc
clear all

mat_dir = fullfile(fileparts(mfilename('fullpath')), 'mat_data');
script_dir = fileparts(mfilename('fullpath'));

load(fullfile(mat_dir, 'Train_data_3Dcase3_261.mat'))
load(fullfile(mat_dir, 'deepOnet_data_3Dcase3_261.mat'))
load(fullfile(mat_dir, 'idx_3Dcase3_261.mat'))

S_test = load(fullfile(mat_dir, 'E_test_pred_size_261_ddp_fft.mat'));
E_pred_test = S_test.E_pred;

Pin = 1.657863990540579E-4;
N = 261;

x = 0.54:0.001:0.8;

% true
nP_R_true = nan(N,1);
nP_T_true = nan(N,1);

% predicted test
nP_R_test_pred = nan(N,1);
nP_T_test_pred = nan(N,1);

%% =========================================================
% 1. Full R_true / T_true from E_true
%    R: boundary 13
%    T: boundary 3
%% =========================================================

% Train set true field
for i = 1:size(Ez_train,1)

    idx = trainIdx(i);
    mesh = Matrix{7,idx};
    P = Matrix{8,idx};

    solver.x = P * Ez_train(i,1:coord_len_train(i)).' / 10;
    phy.k0 = 2*pi/(0.54 + 0.001*(idx-1));

    % R
    triIndex = findTri(13,mesh);
    nP_R_true(idx,1) = get_nP(triIndex,mesh,solver,phy.k0);

    % T
    triIndex = findTri(3,mesh);
    nP_T_true(idx,1) = get_nP(triIndex,mesh,solver,phy.k0);
end

% Test set true field
for i = 1:size(Ez_test,1)

    idx = testIdx(i);
    mesh = Matrix{7,idx};
    P = Matrix{8,idx};

    solver.x = P * Ez_test(i,1:coord_len_test(i)).' / 10;
    phy.k0 = 2*pi/(0.54 + 0.001*(idx-1));

    % R
    triIndex = findTri(13,mesh);
    nP_R_true(idx,1) = get_nP(triIndex,mesh,solver,phy.k0);

    % T
    triIndex = findTri(3,mesh);
    nP_T_true(idx,1) = get_nP(triIndex,mesh,solver,phy.k0);
end

R_true = (Pin + nP_R_true) / Pin;
T_true = nP_T_true / Pin;
A_true = 1 - R_true - T_true;

%% =========================================================
% 2. Test points from predicted field E_pred_test
%% =========================================================

for i = 1:size(Ez_test,1)

    idx = testIdx(i);
    mesh = Matrix{7,idx};
    P = Matrix{8,idx};

    solver.x = P * E_pred_test(i,1:coord_len_test(i)).' / 10;
    phy.k0 = 2*pi/(0.54 + 0.001*(idx-1));

    % R
    triIndex = findTri(13,mesh);
    nP_R_test_pred(idx,1) = get_nP(triIndex,mesh,solver,phy.k0);

    % T
    triIndex = findTri(3,mesh);
    nP_T_test_pred(idx,1) = get_nP(triIndex,mesh,solver,phy.k0);
end

R_test_pred = (Pin + nP_R_test_pred) / Pin;
T_test_pred = nP_T_test_pred / Pin;
A_test_pred = 1 - R_test_pred - T_test_pred;

%% =========================================================
% 3. SCI-style plot: true curves (solid), pred test points (scatter)
%    One scatter every n points; default n = 2
%% =========================================================

% ---------- Figure parameters ----------
figW = 6;     % cm, single-column width
figH = 4;     % cm

lw_true = 1.45;
fs_axis = 8;
fs_label = 9;
fs_legend = 7.5;

% ---------- Scatter parameters ----------
ms_pred = 18;       % marker size
lw_marker = 0.8;    % marker edge width

n_scatter = 2;      % one scatter every n test points; default n = 2
n_scatter = max(1, round(n_scatter));   % ensure n >= 1 and integer

% ---------- Colorblind colors ----------
cR_dark = [0.75, 0.00, 0.00];   % dark red (R true)
cR_lite = [1.00, 0.40, 0.40];   % light red (R pred)

cT_dark = [0.00, 0.30, 0.60];   % dark blue (T true)
cT_lite = [0.30, 0.60, 1.00];   % light blue (T pred)

cA_dark = [0.00, 0.50, 0.30];   % dark green (A true)
cA_lite = [0.40, 0.80, 0.60];   % light green (A pred)

fig = figure('Units', 'centimeters', ...
             'Position', [2 2 figW figH], ...
             'Color', 'w', ...
             'Renderer', 'painters');

ax = axes(fig);
hold(ax, 'on');

% ---------- Test points ----------
testIdx_sort = sort(testIdx(:));

x_test = x(testIdx_sort).';

R_test_pts = R_test_pred(testIdx_sort);
T_test_pts = T_test_pred(testIdx_sort);
A_test_pts = A_test_pred(testIdx_sort);

% True curve values at test lambdas
R_true_test = R_true(testIdx_sort);
T_true_test = T_true(testIdx_sort);
A_true_test = A_true(testIdx_sort);

% ---------- Uniform scatter points near true curves ----------
% Goal: uniform in lambda; per bin pick pred point closest to true curve

n_show_scatter = 15;     % target scatter count; tune e.g. 20, 25, 30, 35

valid_all = isfinite(x_test) & ...
            isfinite(R_test_pts) & isfinite(T_test_pts) & isfinite(A_test_pts) & ...
            isfinite(R_true_test) & isfinite(T_true_test) & isfinite(A_true_test);

cand_id = find(valid_all);

% Cap scatter count if fewer valid test points
n_show_scatter = min(n_show_scatter, numel(cand_id));

% Normalized error so R/A/T scales do not dominate
scale_R = max(max(R_true_test(valid_all)) - min(R_true_test(valid_all)), eps);
scale_T = max(max(T_true_test(valid_all)) - min(T_true_test(valid_all)), eps);
scale_A = max(max(A_true_test(valid_all)) - min(A_true_test(valid_all)), eps);

dist_to_curve = sqrt( ...
    ((R_test_pts - R_true_test) / scale_R).^2 + ...
    ((T_test_pts - T_true_test) / scale_T).^2 + ...
    ((A_test_pts - A_true_test) / scale_A).^2 );

% Uniform bins along lambda
x_cand = x_test(cand_id);
x_edges = linspace(min(x_cand), max(x_cand), n_show_scatter + 1);

scatter_id = nan(n_show_scatter, 1);

for k = 1:n_show_scatter

    if k < n_show_scatter
        in_bin = cand_id(x_test(cand_id) >= x_edges(k) & ...
                         x_test(cand_id) <  x_edges(k+1));
    else
        in_bin = cand_id(x_test(cand_id) >= x_edges(k) & ...
                         x_test(cand_id) <= x_edges(k+1));
    end

    % If bin has test points, pick closest to true curve
    if ~isempty(in_bin)
        [~, local_min_id] = min(dist_to_curve(in_bin));
        scatter_id(k) = in_bin(local_min_id);
    end
end

scatter_id = scatter_id(~isnan(scatter_id));
scatter_id = unique(scatter_id, 'stable');

% Final scatter points for plotting
x_test_scatter = x_test(scatter_id);

R_test_scatter = R_test_pts(scatter_id);
T_test_scatter = T_test_pts(scatter_id);
A_test_scatter = A_test_pts(scatter_id);

valid_R = isfinite(R_test_scatter);
valid_T = isfinite(T_test_scatter);
valid_A = isfinite(A_test_scatter);

% ---------- True curves: solid lines ----------
p1 = plot(ax, x, R_true, '-', ...
    'Color', cR_dark, 'LineWidth', lw_true);

p2 = plot(ax, x, T_true, '-', ...
    'Color', cT_dark, 'LineWidth', lw_true);

p3 = plot(ax, x, A_true, '-', ...
    'Color', cA_dark, 'LineWidth', lw_true);

% ---------- Predicted test points: scatter ----------
p4 = scatter(ax, x_test_scatter(valid_R), R_test_scatter(valid_R), ms_pred, ...
    'Marker', 'o', ...
    'MarkerEdgeColor', cR_lite, ...
    'MarkerFaceColor', cR_lite, ...
    'LineWidth', lw_marker);

p5 = scatter(ax, x_test_scatter(valid_T), T_test_scatter(valid_T), ms_pred, ...
    'Marker', 'o', ...
    'MarkerEdgeColor', cT_lite, ...
    'MarkerFaceColor', cT_lite, ...
    'LineWidth', lw_marker);

p6 = scatter(ax, x_test_scatter(valid_A), A_test_scatter(valid_A), ms_pred, ...
    'Marker', 'o', ...
    'MarkerEdgeColor', cA_lite, ...
    'MarkerFaceColor', cA_lite, ...
    'LineWidth', lw_marker);

% ---------- Axes labels ----------
xlabel(ax, '$\lambda$', ...
    'Interpreter', 'latex', ...
    'FontSize', fs_label);

ylabel(ax, '$R\;/\;T\;/\;A$', ...
    'Interpreter', 'latex', ...
    'FontSize', fs_label);

% ---------- Axes style ----------
set(ax, ...
    'FontName', 'Times New Roman', ...
    'FontSize', fs_axis, ...
    'LineWidth', 0.75, ...
    'Box', 'on', ...
    'TickDir', 'in', ...
    'TickLength', [0.018 0.018], ...
    'XMinorTick', 'on', ...
    'YMinorTick', 'on', ...
    'Layer', 'top');

xlim(ax, [0.54 0.80]);
ylim(ax, [0 1.00]);

xticks(ax, 0.55:0.05:0.80);
yticks(ax, 0:0.2:1.0);

% Avoid displaying 1.0 as 1.000
ax.XAxis.TickLabelFormat = '%.2f';
ax.YAxis.TickLabelFormat = '%.1f';

ax.XMinorTick = 'off';
ax.YMinorTick = 'off';

grid(ax, 'off');

% ---------- Legend ----------
lgd = legend(ax, [p1 p2 p3 p4 p5 p6], ...
    {'$R$ true', '$T$ true', '$A$ true', ...
     '$R$ pred', '$T$ pred', '$A$ pred'}, ...
    'Interpreter', 'latex', ...
    'FontSize', fs_legend, ...
    'Box', 'off', ...
    'NumColumns', 1);

lgd.ItemTokenSize = [18, 8];

% ---------- Manually adjust legend position ----------
lgd.Units = 'normalized';

% [left bottom width height]
% larger left/bottom -> legend further right/up
lgd.Position = [0.6 0.5 0.36 0.18];

% ---------- Tight layout ----------
set(ax, 'Units', 'normalized');
ax.Position = [0.14 0.17 0.80 0.76];

% ---------- Export ----------
set(fig, 'PaperUnits', 'centimeters');
set(fig, 'PaperSize', [figW figH]);
set(fig, 'PaperPosition', [0 0 figW figH]);
set(fig, 'InvertHardcopy', 'off');

outdir = fullfile(script_dir, 'pdf_results');
if ~exist(outdir, 'dir')
    mkdir(outdir);
end

% Recommended: vector PDF
print(fig, fullfile(outdir, 'RTA_1.pdf'), '-dpdf', '-painters', '-r1800');
print(fig, fullfile(outdir, 'RTA_1.svg'), '-dsvg', '-painters');
