clc
clear
close all

mat_dir = fullfile(fileparts(mfilename('fullpath')), 'mat_data');
script_dir = fileparts(mfilename('fullpath'));

addpath(fullfile(script_dir, 'function'));
addpath(fullfile(script_dir, 'kernel'));
addpath(fullfile(script_dir, 'mesh'));
addpath(fullfile(script_dir, 'post'));

load(fullfile(mat_dir, 'Train_data_SPP1_71.mat'))
load(fullfile(mat_dir, 'deepOnet_data_SPP1_71.mat'))
load(fullfile(mat_dir, 'E_test_pred_size_71.mat'))
load(fullfile(mat_dir, 'idx_SPP1_71.mat'))

%% ===================== Global paper plot style =====================
set(groot, 'defaultAxesFontName', 'Times New Roman');
set(groot, 'defaultTextFontName', 'Times New Roman');
set(groot, 'defaultColorbarFontName', 'Times New Roman');
set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');

%% ===================== Select train/test sample =====================
% Match test_loss.m
i = 8;

% Global index of test sample i
sample_id = testIdx(i);

Mesh = Matrix{7, sample_id};
P=Matrix{8,testIdx(i)};

vertex = Mesh.vertex;
tri    = Mesh.tri;

%% ===================== Compute |E| per test_loss.m =====================
Et = P*E_pred(i, 1:coord_len_test(i)).';
[Ex, Ey] = GetExEy(Mesh.vertex, Mesh.tri, Mesh.edgesOfTri, Et);

Et_true = P*Ez_test(i, 1:coord_len_test(i)).';
[Ex_true, Ey_true] = GetExEy(Mesh.vertex, Mesh.tri, Mesh.edgesOfTri, Et_true);

% Same normE definition as test_loss.m
normE = sqrt(abs(Ex).^2 + abs(Ey).^2);
normE_true = sqrt(abs(Ex_true).^2 + abs(Ey_true).^2);

% Residual matches test_loss.m
normE_res = normE_true-normE;

%% ===================== epsilon and Ebz data =====================
eps_real = squeeze(eps_all(sample_id, :, :, 1));

n = 128;
x_range = linspace(-1, 1, n);
y_range = linspace(1, -1, n);
[X, Y] = meshgrid(x_range, y_range);

%% ===================== Color axis limits =====================
% Same color axis for NN and True
c_norm = [min([normE(:); normE_true(:)]), ...
          max([normE(:); normE_true(:)])];

if c_norm(1) == c_norm(2)
    if c_norm(1) == 0
        c_norm = [-1, 1];
    else
        delta = 0.1 * abs(c_norm(1));
        c_norm = [c_norm(1)-delta, c_norm(2)+delta];
    end
end

% Symmetric color axis for residual
max_res = max(abs(normE_res(:)));

if max_res == 0
    c_res = [-1, 1];
else
    c_res = [-max_res, max_res];
end

%% ===================== Colorbar scientific notation exponent =====================
exp_norm = getExponentFromCaxis(c_norm);
exp_res  = getExponentFromCaxis(c_res);

%% ===================== Output folder =====================
outdir = fullfile(script_dir, 'pdf_results');
if ~exist(outdir, 'dir')
    mkdir(outdir);
end

%% =====================================================
%  1) Show 1x3 overview in MATLAB
%  Layout:
%  NN / True / Residual
%% =====================================================
fig_all = figure('Color','w', ...
    'Units','centimeters', ...
    'Position',[2 2 13 4]);

tiledlayout(1, 3, ...
    'TileSpacing','compact', ...
    'Padding','compact');

nexttile;
PlotFieldPaper(vertex, tri, normE, c_norm, exp_norm);
title('NN', 'FontSize', 6, 'FontName', 'Times New Roman');

nexttile;
PlotFieldPaper(vertex, tri, normE_true, c_norm, exp_norm);
title('True', 'FontSize', 6, 'FontName', 'Times New Roman');

nexttile;
PlotFieldPaper(vertex, tri, normE_res, c_res, exp_res);
title('res', 'FontSize', 6, 'FontName', 'Times New Roman');

% Uncomment next line to save 1x3 overview
% exportgraphics(fig_all, fullfile(outdir, ['A', num2str(i), '_E_norm_1x3.pdf']), 'ContentType','vector');

%% =====================================================
%  2) Save 3 separate |E| PDFs
%% =====================================================
saveSinglePDF(fullfile(outdir, ['SPP1_', num2str(i), '_E_norm_pred.pdf']), ...
    vertex, tri, normE, c_norm, exp_norm);

saveSinglePDF(fullfile(outdir, ['SPP1_', num2str(i), '_E_norm_true.pdf']), ...
    vertex, tri, normE_true, c_norm, exp_norm);

saveSinglePDF(fullfile(outdir, ['SPP1_', num2str(i), '_E_norm_res.pdf']), ...
    vertex, tri, normE_res, c_res, exp_res);

%% =====================================================
%  3) Save epsilon and Ebz real-part PDF
%% =====================================================
eps_file = fullfile(outdir, ['SPP1_', num2str(i), '_epsilon_real.pdf']);

saveEpsilonPDF(eps_file, X, Y, eps_real);

disp(['3 |E|-field PDFs saved to: ', outdir]);
disp(['Epsilon (real part) PDF saved: ', eps_file]);


%% =====================================================
%  Local functions
%% =====================================================

function saveSinglePDF(filename, node, elem, Ez, cax, expPow)

figW = 2.5;
figH = 3;
% Single figure size: 4.2 cm x 3 cm
fig = figure('Color','w', ...
    'Units','centimeters', ...
    'Position',[2 2 figW figH], ...
    'Visible','off');

PlotFieldPaper(node, elem, Ez, cax, expPow);

set(fig, 'PaperUnits','centimeters');
set(fig, 'PaperPosition',[0 0 figW figH]);
set(fig, 'PaperSize',[figW figH]);

exportgraphics(fig, filename, ...
    'ContentType','vector', ...
    'BackgroundColor','white');

close(fig);

end


function saveEpsilonPDF(filename, X, Y, eps_real)
% epsilon: keep output_Eps_Ebz size and no-colorbar style
fig = figure('Color','w', ...
    'Units','centimeters', ...
    'Position',[2 2 3.0 3.0], ...
    'Visible','off');

PlotEpsilonMaterial(fig, X, Y, eps_real);

set(fig, 'PaperUnits', 'centimeters');
set(fig, 'PaperPosition', [0 0 3.0 3.0]);
set(fig, 'PaperSize', [3.0 3.0]);

exportgraphics(fig, filename, ...
    'ContentType', 'vector', ...
    'BackgroundColor', 'white');

close(fig);
end


function saveEbzPDF(filename, X, Y, Ebz_real)
% Ebz figure size matches outputE_v2 saveSinglePDF,
% so default colorbar layout matches outputE_v2.
fig = figure('Color','w', ...
    'Units','centimeters', ...
    'Position',[2 2 4.2 4], ...
    'Visible','off');

PlotEbzPaper(X, Y, Ebz_real);

set(fig, 'PaperUnits','centimeters');
set(fig, 'PaperPosition',[0 0 4.2 3]);
set(fig, 'PaperSize',[4.2 3]);

exportgraphics(fig, filename, ...
    'ContentType','vector', ...
    'BackgroundColor','white');

close(fig);
end


function PlotFieldPaper(node, elem, Ez, cax, expPow)

% Scale display only; caxis unchanged
scale = 10^expPow;

Ez_plot  = Ez  / scale;
cax_plot = cax / scale;

% 2D pseudocolor field plot
trisurf(elem, ...
    node(:,1), node(:,2), zeros(size(Ez_plot)), Ez_plot, ...
    'EdgeColor','none');

shading interp
view(2)

axis equal
axis tight
axis off

colormap(jet)
caxis(cax_plot)

% colorbar settings
cb = colorbar;
cb.FontName = 'Times New Roman';
cb.FontSize = 6;
cb.TickLabelInterpreter = 'latex';
cb.LineWidth = 0.25;
cb.Box = 'on';

% colorbar: one significant digit
[ticks, ticklabels] = getOneSigTicks(cax_plot);
cb.Ticks = ticks;
cb.TickLabels = ticklabels;

% colorbar top: e.g. x10^{-2}
if expPow ~= 0
    title(cb, ['$\times 10^{', num2str(expPow), '}$'], ...
        'Interpreter','latex', ...
        'FontSize',6, ...
        'FontName','Times New Roman');
else
    title(cb, '', 'Interpreter','latex');
end

set(gca, ...
    'FontName','Times New Roman', ...
    'LineWidth',0.8);

end


function PlotEpsilonMaterial(fig, X, Y, eps_real)
% =====================================================
% epsilon real part as material map:
% air = light gray
% silicon = dark gray
% no colorbar
% Same style as output_Eps_Ebz.
% =====================================================

ax = axes('Parent', fig, ...
    'Units', 'normalized', ...
    'Position', [0.02 0.02 0.96 0.96]);

% Threshold at midpoint for two materials
e_min = min(eps_real(:));
e_max = max(eps_real(:));
th = 0.5 * (e_min + e_max);

material_mask = eps_real > th;   % 0: air, 1: silicon

imagesc(ax, X(1,:), Y(:,1), material_mask);
set(ax, 'YDir', 'normal');

axis(ax, 'equal');
axis(ax, 'tight');
axis(ax, 'off');

% Colors: air light gray, silicon dark gray
airColor = [0.90, 0.90, 0.90];
siColor  = [0.35, 0.35, 0.35];

colormap(ax, [airColor; siColor]);
caxis(ax, [0 1]);

ax.FontName = 'Times New Roman';
ax.LineWidth = 0.8;
end


function PlotEbzPaper(X, Y, F)
% =====================================================
% Ebz real: colorbar style matches outputE_v2
% PlotFieldPaper colorbar.
% Use MATLAB default ax/cb positions (do not set manually),
% matching font/size/latex/linewidth/box.
% =====================================================

cax = [min(F(:)), max(F(:))];

% Avoid degenerate caxis for constant fields
if cax(1) == cax(2)
    if cax(1) == 0
        cax = [-1, 1];
    else
        delta = 0.1 * abs(cax(1));
        cax = [cax(1)-delta, cax(2)+delta];
    end
end

expPow = getExponentFromCaxis(cax);

% Scale display only; caxis unchanged
scale = 10^expPow;
F_plot  = F   / scale;
cax_plot = cax / scale;

imagesc(X(1,:), Y(:,1), F_plot);
set(gca, 'YDir', 'normal');

axis equal
axis tight
axis off

colormap(jet)
caxis(cax_plot)

% colorbar settings: same as PlotFieldPaper
cb = colorbar;
cb.FontName = 'Times New Roman';
cb.FontSize = 6;
cb.TickLabelInterpreter = 'latex';
cb.LineWidth = 0.5;
cb.Box = 'on';

% One significant digit on colorbar, as PlotFieldPaper
[ticks, ticklabels] = getOneSigTicks(cax_plot);
cb.Ticks = ticks;
cb.TickLabels = ticklabels;

% colorbar top: e.g. x10^{-2}
if expPow ~= 0
    title(cb, ['$\times 10^{', num2str(expPow), '}$'], ...
        'Interpreter','latex', ...
        'FontSize',6, ...
        'FontName','Times New Roman');
else
    title(cb, '', 'Interpreter','latex');
end

set(gca, ...
    'FontName','Times New Roman', ...
    'LineWidth',0.8);
end


function expPow = getExponentFromCaxis(cax)

vmax = max(abs(cax(:)));

if vmax == 0
    expPow = 0;
    return;
end

expPow = floor(log10(vmax));

% No x10^n when values are between 10^{-1} and 10^{1}
if abs(expPow) <= 1
    expPow = 0;
end

end


function [ticks, ticklabels] = getOneSigTicks(cax_plot)

a = cax_plot(1);
b = cax_plot(2);

if a == b
    ticks = a;
    ticklabels = {formatOneSig(a)};
    return;
end

% Target ~4-5 ticks
targetNum = 5;

rawStep = (b - a) / (targetNum - 1);
step = getNiceStep(rawStep);

% Pick nice ticks within caxis
t1 = ceil(a / step) * step;
t2 = floor(b / step) * step;

ticks = t1 : step : t2;

% Include 0 on colorbar if range crosses zero
if a < 0 && b > 0
    if ~any(abs(ticks) < 1e-12)
        ticks = sort([ticks, 0]);
    end
end

% Remove floating-point noise
ticks(abs(ticks) < 1e-12) = 0;
ticks = unique(round(ticks, 12), 'stable');

ticklabels = arrayfun(@(x) formatOneSig(x), ...
    ticks, ...
    'UniformOutput', false);

end


function step = getNiceStep(rawStep)

if rawStep <= 0 || ~isfinite(rawStep)
    step = 1;
    return;
end

p = floor(log10(abs(rawStep)));
base = 10^p;

% Candidate steps; factor 4 yields ticks like 0.4, 0, -0.4, -0.8
candidates = [1, 2, 4, 5, 10] * base;

[~, idx] = min(abs(candidates - rawStep));
step = candidates(idx);

end


function s = formatOneSig(x)

if abs(x) < 1e-12
    s = '0';
else
    s = sprintf('%.2g', x);
end

end
