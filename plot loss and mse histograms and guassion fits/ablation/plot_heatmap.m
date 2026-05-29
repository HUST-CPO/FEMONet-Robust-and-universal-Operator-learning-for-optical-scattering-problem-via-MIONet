%% Ablation heatmap: best_test_fem_loss vs learning rate and network size.
clear; clc; close all;

P = get_plot_paths();
csvFile = fullfile(P.ablation_dir, 'ablation_grid_progress.csv');

if ~isfile(csvFile)
    files = dir(fullfile(P.ablation_dir, 'ablation_grid_progress*.csv'));
    if isempty(files)
        error('CSV not found in: %s', P.ablation_dir);
    end
    csvFile = fullfile(P.ablation_dir, files(1).name);
end

T = readtable(csvFile);

learningRates = [1e-4, 3e-4, 5e-4, 1e-3, 3e-3, 5e-3, 1e-2];
lrLabels = {'1e-4','3e-4','5e-4','1e-3','3e-3','5e-3','1e-2'};
outputDims = [64, 128, 256];
hiddenChannels = [128, 256, 512];

varNames = T.Properties.VariableNames;
if ismember('hidden_channel', varNames)
    hiddenCol = 'hidden_channel';
elseif ismember('diden_channel', varNames)
    hiddenCol = 'diden_channel';
else
    error('CSV must contain hidden_channel or diden_channel column.');
end

requiredCols = {'output_dim', 'learning_rate', 'best_test_fem_loss'};
for i = 1:numel(requiredCols)
    if ~ismember(requiredCols{i}, varNames)
        error('Missing column: %s', requiredCols{i});
    end
end

if ismember('status', varNames)
    T = T(strcmp(string(T.status), 'SUCCESS'), :);
end

nRows = numel(outputDims) * numel(hiddenChannels);
nCols = numel(learningRates);
Z = nan(nRows, nCols);
yLabels = strings(nRows, 1);

rowIdx = 0;
for i = 1:numel(outputDims)
    for j = 1:numel(hiddenChannels)
        rowIdx = rowIdx + 1;
        od = outputDims(i);
        hc = hiddenChannels(j);
        yLabels(rowIdx) = sprintf('od=%d, hc=%d', od, hc);
        for k = 1:numel(learningRates)
            lr = learningRates(k);
            idx = T.output_dim == od & T.(hiddenCol) == hc & abs(T.learning_rate - lr) < 1e-12;
            if any(idx)
                vals = T.best_test_fem_loss(idx);
                vals = vals(~isnan(vals));
                if ~isempty(vals)
                    Z(rowIdx, k) = min(vals);
                end
            end
        end
    end
end

fig = figure('Color', 'w', 'Position', [100, 100, 1050, 600]);
Zplot = Z;
Zplot(Zplot <= 0) = NaN;
Zlog = log10(Zplot);
imagesc(Zlog, 'AlphaData', ~isnan(Zlog));
set(gca, 'Color', [0.85, 0.85, 0.85]);
colormap(hot);
cb = colorbar;
caxis([-5, -2]);
cb.Ticks = [-5, -4, -3, -2];
cb.TickLabels = {'1e-5', '1e-4', '1e-3', '1e-2'};
cb.Label.String = 'best\_test\_fem\_loss';
xticks(1:nCols);
xticklabels(lrLabels);
yticks(1:nRows);
yticklabels(cellstr(yLabels));
xlabel('Learning rate');
ylabel('output\_dim, hidden\_channel');
title('Ablation Grid Heatmap: best\_test\_fem\_loss');
set(gca, 'YDir', 'reverse');

for r = 1:nRows
    for c = 1:nCols
        if ~isnan(Zplot(r, c))
            lbl = sprintf('%.2e', Zplot(r, c));
        else
            lbl = 'NA';
        end
        text(c, r, lbl, 'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'middle', 'FontSize', 9, 'Color', 'k');
    end
end
box on;

[minVal, linearIdx] = min(Z(:), [], 'omitnan');
[bestRow, bestCol] = ind2sub(size(Z), linearIdx);
fprintf('Best best_test_fem_loss = %.8g\n', minVal);
fprintf('Best: %s, lr = %s\n', yLabels(bestRow), lrLabels{bestCol});

pdf_path = fullfile(P.out_dir, 'best_test_fem_loss_heatmap.pdf');
svg_path = fullfile(P.out_dir, 'best_test_fem_loss_heatmap.svg');
exportgraphics(fig, pdf_path, 'ContentType', 'vector');
saveas(fig, svg_path);
fprintf('Saved: %s\n', pdf_path);
