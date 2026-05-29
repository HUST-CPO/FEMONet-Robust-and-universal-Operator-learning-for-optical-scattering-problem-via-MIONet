function plot_mse_gaussian(case_id, x_log_range, out_name)
%PLOT_MSE_GAUSSIAN  Log-spaced MSE histogram with Gaussian fit in log10 space.
%   Reads mse_cache/<case_id>_train.mat and _test.mat
%   x_log_range: [log10_min log10_max], default [-6 0]

    if nargin < 2 || isempty(x_log_range)
        x_log_range = [-6, 0];
    end
    if nargin < 3 || isempty(out_name)
        out_name = ['error_MSE_', case_id];
    end

    P = get_plot_paths();
    St = load(fullfile(P.mse_dir, [case_id, '_test.mat']));
    Sr = load(fullfile(P.mse_dir, [case_id, '_train.mat']));

    train_err = Sr.sample_mse(:);
    test_err = St.sample_mse(:);
    train_err = train_err(isfinite(train_err) & train_err > 0);
    test_err = test_err(isfinite(test_err) & test_err > 0);

    z_train = log10(train_err);
    z_test = log10(test_err);
    mu_tr = mean(z_train);  sig_tr = std(z_train);
    mu_te = mean(z_test);   sig_te = std(z_test);

    x_min_plot = 10^x_log_range(1);
    x_max_plot = 10^x_log_range(2);
    nbins = 50;
    edges = logspace(log10(x_min_plot), log10(x_max_plot), nbins + 1);
    bin_centers = sqrt(edges(1:end-1) .* edges(2:end));

    train_fit = normcdf(log10(edges(2:end)), mu_tr, sig_tr) - ...
                normcdf(log10(edges(1:end-1)), mu_tr, sig_tr);
    test_fit = normcdf(log10(edges(2:end)), mu_te, sig_te) - ...
               normcdf(log10(edges(1:end-1)), mu_te, sig_te);

    set(groot, 'defaultAxesFontName', 'Times New Roman');
    set(groot, 'defaultTextFontName', 'Times New Roman');
    set(groot, 'defaultLegendFontName', 'Times New Roman');

    fig = figure('Color', 'w', 'Units', 'centimeters', 'Position', [10 10 5 4]);
    hold on;

    h1 = histogram(train_err, edges, 'Normalization', 'probability', ...
        'FaceColor', [0.3 0.7 1.0], 'EdgeColor', 'b', 'FaceAlpha', 0.30, 'LineWidth', 0.5);
    h2 = histogram(test_err, edges, 'Normalization', 'probability', ...
        'FaceColor', [1.0 0.6 0.5], 'EdgeColor', 'r', 'FaceAlpha', 0.30, 'LineWidth', 0.5);
    p1 = plot(bin_centers, train_fit, 'b-', 'LineWidth', 0.5);
    p2 = plot(bin_centers, test_fit, 'r-', 'LineWidth', 0.5);

    set(gca, 'XScale', 'log', 'FontSize', 7, 'LineWidth', 0.5, ...
        'FontName', 'Times New Roman', 'YMinorTick', 'off', 'XMinorTick', 'off');
    xlabel('MSE Error', 'FontSize', 7);
    ylabel('Proportion', 'FontSize', 7);
    lgd = legend([h1 h2 p1 p2], {'Train Data', 'Test Data', 'Train Fit', 'Test Fit'}, ...
        'Box', 'off', 'FontSize', 7);
    lgd.ItemTokenSize = [8, 4];
    lgd.Units = 'normalized';
    lgd.Position = [0.59, 0.63, 0.30, 0.22];
    box on;
    xlim([x_min_plot, x_max_plot]);
    max_y = max([h1.Values, h2.Values, train_fit, test_fit]);
    ylim([0, max_y * 1.15]);
    ytickformat('%.2f');

    pdf_path = fullfile(P.out_dir, [out_name, '.pdf']);
    exportgraphics(fig, pdf_path, 'ContentType', 'vector');
    fprintf('Saved MSE histogram: %s\n', pdf_path);
    fprintf('Train log10: mu=%.4g sigma=%.4g | Test log10: mu=%.4g sigma=%.4g\n', ...
        mu_tr, sig_tr, mu_te, sig_te);
end
