clc
clear
close all

mat_dir = fullfile(fileparts(mfilename('fullpath')), 'mat_data');
script_dir = fileparts(mfilename('fullpath'));

load(fullfile(mat_dir, 'Train_data_A_1558.mat'))
load(fullfile(mat_dir, 'deepOnet_data_A_1558.mat'))
load(fullfile(mat_dir, 'E_test_pred_size_1558.mat'))
load(fullfile(mat_dir, 'idx_A_1558.mat'))

%% ===================== Global paper plot style =====================
set(groot, 'defaultAxesFontName', 'Times New Roman');
set(groot, 'defaultTextFontName', 'Times New Roman');
set(groot, 'defaultColorbarFontName', 'Times New Roman');
set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');

%% ===================== Select test sample =====================
i = 273;

sample_id = testIdx(1,i);

vertex = Matrix{6, sample_id} / 1e6;
tri    = Matrix{7, sample_id};
E_true = Matrix{5, sample_id};

E_pred_i = E_pred(i, 1:coord_len_test(i)).';

%% ===================== Decompose field data =====================
E_pred_real = real(E_pred_i);
E_pred_imag = imag(E_pred_i);

E_true_real = real(E_true);
E_true_imag = imag(E_true);

E_res_real = real(E_true - E_pred_i);
E_res_imag = imag(E_true - E_pred_i);

%% ===================== epsilon and Ebz data =====================
eps_real = squeeze(eps_all(sample_id, :, :, 1));

n = 128;
x_range = linspace(-1, 1, n);
y_range = linspace(1, -1, n);
[X, Y] = meshgrid(x_range, y_range);

%% ===================== Color axis limits =====================
c_real = [min([E_pred_real(:); E_true_real(:)]), ...
          max([E_pred_real(:); E_true_real(:)])];

c_imag = [min([E_pred_imag(:); E_true_imag(:)]), ...
          max([E_pred_imag(:); E_true_imag(:)])];

max_res_real = max(abs(E_res_real(:)));
max_res_imag = max(abs(E_res_imag(:)));

if max_res_real == 0
    c_res_real = [-1, 1];
else
    c_res_real = [-max_res_real, max_res_real];
end

if max_res_imag == 0
    c_res_imag = [-1, 1];
else
    c_res_imag = [-max_res_imag, max_res_imag];
end

%% ===================== Colorbar scientific-notation exponent =====================
exp_real     = getExponentFromCaxis(c_real);
exp_imag     = getExponentFromCaxis(c_imag);
exp_res_real = getExponentFromCaxis(c_res_real);
exp_res_imag = getExponentFromCaxis(c_res_imag);

%% ===================== Output directory =====================
outdir = fullfile(script_dir, 'pdf_results');
if ~exist(outdir, 'dir')
    mkdir(outdir);
end

%% =====================================================
%  1) Display 2x3 overview in MATLAB
%% =====================================================
fig_all = figure('Color','w', ...
    'Units','centimeters', ...
    'Position',[2 2 24 13]);

tiledlayout(2, 3, ...
    'TileSpacing','compact', ...
    'Padding','compact');

nexttile;
PlotFieldPaperTile(vertex, tri, E_pred_real, c_real, exp_real);

nexttile;
PlotFieldPaperTile(vertex, tri, E_true_real, c_real, exp_real);

nexttile;
PlotFieldPaperTile(vertex, tri, E_res_real, c_res_real, exp_res_real);

nexttile;
PlotFieldPaperTile(vertex, tri, E_pred_imag, c_imag, exp_imag);

nexttile;
PlotFieldPaperTile(vertex, tri, E_true_imag, c_imag, exp_imag);

nexttile;
PlotFieldPaperTile(vertex, tri, E_res_imag, c_res_imag, exp_res_imag);

%% =====================================================
%  2) Export 6 field plots
%
%  PDF: no colorbar, filenames *_nc.pdf
%  SVG: with colorbar, filenames *.svg
%% =====================================================

saveFieldPDFNoColorbar_SVGColorbar( ...
    fullfile(outdir, ['A', num2str(i), '_pred_real_nc.pdf']), ...
    fullfile(outdir, ['A', num2str(i), '_pred_real.svg']), ...
    vertex, tri, E_pred_real, c_real, exp_real);

saveFieldPDFNoColorbar_SVGColorbar( ...
    fullfile(outdir, ['A', num2str(i), '_true_real_nc.pdf']), ...
    fullfile(outdir, ['A', num2str(i), '_true_real.svg']), ...
    vertex, tri, E_true_real, c_real, exp_real);

saveFieldPDFNoColorbar_SVGColorbar( ...
    fullfile(outdir, ['A', num2str(i), '_res_real_nc.pdf']), ...
    fullfile(outdir, ['A', num2str(i), '_res_real.svg']), ...
    vertex, tri, E_res_real, c_res_real, exp_res_real);

saveFieldPDFNoColorbar_SVGColorbar( ...
    fullfile(outdir, ['A', num2str(i), '_pred_imag_nc.pdf']), ...
    fullfile(outdir, ['A', num2str(i), '_pred_imag.svg']), ...
    vertex, tri, E_pred_imag, c_imag, exp_imag);

saveFieldPDFNoColorbar_SVGColorbar( ...
    fullfile(outdir, ['A', num2str(i), '_true_imag_nc.pdf']), ...
    fullfile(outdir, ['A', num2str(i), '_true_imag.svg']), ...
    vertex, tri, E_true_imag, c_imag, exp_imag);

saveFieldPDFNoColorbar_SVGColorbar( ...
    fullfile(outdir, ['A', num2str(i), '_res_imag_nc.pdf']), ...
    fullfile(outdir, ['A', num2str(i), '_res_imag.svg']), ...
    vertex, tri, E_res_imag, c_res_imag, exp_res_imag);

%% =====================================================
%  3) Save epsilon real-part PDF/SVG
%% =====================================================
eps_file = fullfile(outdir, ['A', num2str(i), '_epsilon_real.pdf']);
saveEpsilonPDF(eps_file, X, Y, eps_real);


disp(['6 E-field PDFs (no colorbar) saved to: ', outdir]);
disp(['6 E-field SVGs (with colorbar) saved to: ', outdir]);
disp(['Epsilon (real part) PDF/SVG saved: ', eps_file]);

%% =====================================================
%  Local functions
%% =====================================================

function saveFieldPDFNoColorbar_SVGColorbar(pdfname_nc, svgname, node, elem, Ez, cax, expPow)
% =====================================================
% PDF: no colorbar, field only, filename *_nc.pdf
% SVG: with colorbar, filename *.svg
% =====================================================

figW = 2.1;
figH = 1.5;

%% ---------- 1) Save PDF without colorbar ----------
fig_pdf = figure('Color','w', ...
    'Units','centimeters', ...
    'Position',[2 2 2 2], ...
    'Visible','off');

PlotFieldPaperSingleNoColorbar(fig_pdf, node, elem, Ez, cax, expPow);

set(fig_pdf, 'PaperUnits','centimeters');
set(fig_pdf, 'PaperPosition',[0 0 2 2]);
set(fig_pdf, 'PaperSize',[2 2]);
set(fig_pdf, 'PaperPositionMode','manual');
set(fig_pdf, 'InvertHardcopy','off');
set(fig_pdf, 'Renderer','painters');

print(fig_pdf, pdfname_nc, '-dpdf', '-painters');

close(fig_pdf);

%% ---------- 2) Save SVG with colorbar ----------
fig_svg = figure('Color','w', ...
    'Units','centimeters', ...
    'Position',[2 2 figW figH], ...
    'Visible','off');

PlotFieldPaperSingle(fig_svg, node, elem, Ez, cax, expPow);

set(fig_svg, 'PaperUnits','centimeters');
set(fig_svg, 'PaperPosition',[0 0 figW figH]);
set(fig_svg, 'PaperSize',[figW figH]);
set(fig_svg, 'PaperPositionMode','manual');
set(fig_svg, 'InvertHardcopy','off');
set(fig_svg, 'Renderer','painters');

print(fig_svg, svgname, '-dsvg', '-painters');

close(fig_svg);

end


function saveEpsilonPDF(filename, X, Y, eps_real)

figW = 3.0;
figH = 3.0;

fig = figure('Color','w', ...
    'Units','centimeters', ...
    'Position',[2 2 figW figH], ...
    'Visible','off');

PlotEpsilonMaterial(fig, X, Y, eps_real);

set(fig, 'PaperUnits', 'centimeters');
set(fig, 'PaperPosition', [0 0 figW figH]);
set(fig, 'PaperSize', [figW figH]);
set(fig, 'PaperPositionMode','manual');
set(fig, 'InvertHardcopy','off');

saveFigurePDFSVG(fig, filename);

close(fig);

end

function saveEbzNoColorbarPDFOnly(filename, X, Y, Ebz_real)
% =====================================================
% Ebz real part: single PDF without colorbar
%
% Example output:
%   C2_Ebz_real.pdf
% =====================================================

figW = 2.1;
figH = 1.5;

fig = figure('Color','w', ...
    'Units','centimeters', ...
    'Position',[2 2 figW figH], ...
    'Visible','off');

PlotEbzPaperSingleNoColorbar(fig, X, Y, Ebz_real);

set(fig, 'PaperUnits','centimeters');
set(fig, 'PaperPosition',[0 0 2 2]);
set(fig, 'PaperSize',[2 2]);
set(fig, 'PaperPositionMode','manual');
set(fig, 'InvertHardcopy','off');
set(fig, 'Renderer','painters');

% PDF only, no SVG
print(fig, filename, '-dpdf', '-painters');

close(fig);

end

function PlotEbzPaperSingleNoColorbar(fig, X, Y, F)
% =====================================================
% Standalone Ebz export: no colorbar, field only
% =====================================================

cax = [min(F(:)), max(F(:))];

if cax(1) == cax(2)
    if cax(1) == 0
        cax = [-1, 1];
    else
        delta = 0.1 * abs(cax(1));
        cax = [cax(1)-delta, cax(2)+delta];
    end
end

expPow = getExponentFromCaxis(cax);

scale = 10^expPow;
F_plot  = F / scale;
cax_plot = cax / scale;

% No colorbar; field fills canvas
axPos = [0 0 1 1];

ax = axes('Parent', fig, ...
    'Units','normalized', ...
    'Position', axPos);

imagesc(ax, X(1,:), Y(:,1), F_plot);
set(ax, 'YDir', 'normal');

axis(ax, 'equal');
axis(ax, 'tight');
axis(ax, 'off');

colormap(ax, jet(256));
caxis(ax, cax_plot);

set(ax, ...
    'Units','normalized', ...
    'Position', axPos, ...
    'FontName','Times New Roman', ...
    'LineWidth',0.8);

end

function saveFigurePDFSVG(fig, pdfname)

svgname = pdf2svgName(pdfname);

set(fig, 'Renderer', 'painters');

print(fig, pdfname, '-dpdf', '-painters');
print(fig, svgname, '-dsvg', '-painters');

end


function svgname = pdf2svgName(pdfname)

[p, n, ~] = fileparts(pdfname);
svgname = fullfile(p, [n, '.svg']);

end


function PlotFieldPaperTile(node, elem, Ez, cax, expPow)

scale = 10^expPow;

Ez_plot  = Ez(:)  / scale;
cax_plot = cax / scale;

trisurf(elem, ...
    node(:,1), node(:,2), zeros(size(node(:,1))), Ez_plot, ...
    'EdgeColor','none');

shading interp
view(2)

axis equal
axis tight
axis off

colormap(jet(256))
caxis(cax_plot)

cb = colorbar;
cb.FontName = 'Times New Roman';
cb.FontSize = 6;
cb.TickLabelInterpreter = 'latex';
cb.LineWidth = 0.25;
cb.Box = 'on';

[ticks, ticklabels] = getOneSigTicks(cax_plot);
cb.Ticks = ticks;
cb.TickLabels = ticklabels;

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


function PlotFieldPaperSingle(fig, node, elem, Ez, cax, expPow)
% =====================================================
% For SVG: with colorbar
% =====================================================

scale = 10^expPow;

Ez_plot  = Ez(:)  / scale;
cax_plot = cax / scale;

axPos = [0 0 0.66 0.82];
cbPos = [0.7 0 0.08 0.82];

ax = axes('Parent', fig, ...
    'Units','normalized', ...
    'Position', axPos);

axes(ax);

trisurf(elem, ...
    node(:,1), node(:,2), zeros(size(node(:,1))), Ez_plot, ...
    'EdgeColor','none');

shading interp
view(2)

axis equal
axis tight
axis off

colormap(jet(256))
caxis(cax_plot)

set(ax, ...
    'FontName','Times New Roman', ...
    'LineWidth',0.8);

cb = colorbar;

set(ax, 'Units','normalized', 'Position', axPos);
set(cb, 'Units','normalized', 'Position', cbPos);

cb.FontName = 'Times New Roman';
cb.FontSize = 6;
cb.TickLabelInterpreter = 'latex';
cb.LineWidth = 0.25;
cb.Box = 'on';

[ticks, ticklabels] = getOneSigTicks(cax_plot);
cb.Ticks = ticks;
cb.TickLabels = ticklabels;

if expPow ~= 0
    annotation(fig, 'textbox', ...
        [0.68 0.84 0.20 0.14], ...
        'String', ['$\times 10^{', num2str(expPow), '}$'], ...
        'Interpreter','latex', ...
        'FontName','Times New Roman', ...
        'FontSize',4, ...
        'LineStyle','none', ...
        'HorizontalAlignment','center', ...
        'VerticalAlignment','middle', ...
        'FitBoxToText','off');
end

end


function PlotFieldPaperSingleNoColorbar(fig, node, elem, Ez, cax, expPow)
% =====================================================
% For PDF: no colorbar, field only
% =====================================================

scale = 10^expPow;

Ez_plot  = Ez(:)  / scale;
cax_plot = cax / scale;

axPos = [0 0 1 1];

ax = axes('Parent', fig, ...
    'Units','normalized', ...
    'Position', axPos);

axes(ax);

trisurf(elem, ...
    node(:,1), node(:,2), zeros(size(node(:,1))), Ez_plot, ...
    'EdgeColor','none');

shading interp
view(2)

axis equal
axis tight
axis off

colormap(jet(256))
caxis(cax_plot)

set(ax, ...
    'Units','normalized', ...
    'Position', axPos, ...
    'FontName','Times New Roman', ...
    'LineWidth',0.8);

end


function PlotEpsilonMaterial(fig, X, Y, eps_real)

ax = axes('Parent', fig, ...
    'Units', 'normalized', ...
    'Position', [0.02 0.02 0.96 0.96]);

e_min = min(eps_real(:));
e_max = max(eps_real(:));
th = 0.5 * (e_min + e_max);

material_mask = eps_real > th;

imagesc(ax, X(1,:), Y(:,1), material_mask);
set(ax, 'YDir', 'normal');

axis(ax, 'equal');
axis(ax, 'tight');
axis(ax, 'off');

airColor = [0.90, 0.90, 0.90];
siColor  = [0.35, 0.35, 0.35];

colormap(ax, [airColor; siColor]);
caxis(ax, [0 1]);

ax.FontName = 'Times New Roman';
ax.LineWidth = 0.8;

end


function PlotEbzPaperSingle(fig, X, Y, F)

cax = [min(F(:)), max(F(:))];

if cax(1) == cax(2)
    if cax(1) == 0
        cax = [-1, 1];
    else
        delta = 0.1 * abs(cax(1));
        cax = [cax(1)-delta, cax(2)+delta];
    end
end

expPow = getExponentFromCaxis(cax);

scale = 10^expPow;
F_plot  = F / scale;
cax_plot = cax / scale;

axPos = [0.08 0.08 0.66 0.82];
cbPos = [0.82 0.15 0.045 0.68];

ax = axes('Parent', fig, ...
    'Units','normalized', ...
    'Position', axPos);

imagesc(ax, X(1,:), Y(:,1), F_plot);
set(ax, 'YDir', 'normal');

axis(ax, 'equal');
axis(ax, 'tight');
axis(ax, 'off');

colormap(ax, jet(256));
caxis(ax, cax_plot);

cb = colorbar(ax);

set(ax, 'Units','normalized', 'Position', axPos);
set(cb, 'Units','normalized', 'Position', cbPos);

cb.FontName = 'Times New Roman';
cb.FontSize = 6;
cb.TickLabelInterpreter = 'latex';
cb.LineWidth = 0.25;
cb.Box = 'on';

[ticks, ticklabels] = getOneSigTicks(cax_plot);
cb.Ticks = ticks;
cb.TickLabels = ticklabels;

if expPow ~= 0
    annotation(fig, 'textbox', ...
        [0.72 0.82 0.26 0.14], ...
        'String', ['$\times 10^{', num2str(expPow), '}$'], ...
        'Interpreter','latex', ...
        'FontName','Times New Roman', ...
        'FontSize',6, ...
        'LineStyle','none', ...
        'HorizontalAlignment','center', ...
        'VerticalAlignment','middle', ...
        'FitBoxToText','off');
end

set(ax, ...
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

targetNum = 5;

rawStep = (b - a) / (targetNum - 1);
step = getNiceStep(rawStep);

t1 = ceil(a / step) * step;
t2 = floor(b / step) * step;

ticks = t1 : step : t2;

if a < 0 && b > 0
    if ~any(abs(ticks) < 1e-12)
        ticks = sort([ticks, 0]);
    end
end

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

candidates = [1, 2, 4, 5, 10] * base;

[~, idx] = min(abs(candidates - rawStep));
step = candidates(idx);

end


function s = formatOneSig(x)

if abs(x) < 1e-12
    s = '0';
else
    s = sprintf('%.1g', x);
end

end