function plot_loss_modelAV()
%PLOT_LOSS_MODELAV  Case A FEM loss vs ablation MSE (modelAV CSV).

    P = get_plot_paths();

    logfile = fullfile(P.log_dir, 'modelA_loss.txt');
    txt = fileread(logfile);
    pattern = 'Epoch\s+(\d+),\s+Total Loss:\s*([0-9eE\.\-\+]+),\s*test Loss\s*([0-9eE\.\-\+]+)';
    tokens = regexp(txt, pattern, 'tokens');
    n = numel(tokens);
    epochs = zeros(n, 1);
    train_loss = zeros(n, 1);
    test_loss = zeros(n, 1);
    for k = 1:n
        epochs(k) = str2double(tokens{k}{1});
        train_loss(k) = str2double(tokens{k}{2});
        test_loss(k) = str2double(tokens{k}{3});
    end

    csvfile = fullfile(P.log_dir, 'modelAV_loss.csv');
    if ~isfile(csvfile)
        alt = dir(fullfile(P.log_dir, 'modelAV_loss*.csv'));
        if isempty(alt)
            error('modelAV_loss.csv not found in %s', P.log_dir);
        end
        csvfile = fullfile(P.log_dir, alt(1).name);
    end
    T = readtable(csvfile);

    color_light_blue = [0.35, 0.65, 0.95];
    color_light_red  = [0.95, 0.45, 0.45];

    fig = figure('Color', 'w', 'Units', 'centimeters', 'Position', [10 10 5 4]);
    semilogy(epochs, train_loss, 'b-', 'LineWidth', 0.5); hold on;
    semilogy(epochs, test_loss, 'r-', 'LineWidth', 0.5);
    semilogy(T.epoch, T.mse_train, '-', 'Color', color_light_blue, 'LineWidth', 0.5);
    semilogy(T.epoch, T.mse_test, '-', 'Color', color_light_red, 'LineWidth', 0.5);

    xlabel('Epoch');
    ylabel('MSE');
    lgd = legend('Train', 'Test', 'V-Train', 'V-Test', 'Location', 'northeast', 'Box', 'off');
    lgd.FontName = 'Times New Roman';
    lgd.ItemTokenSize = [8, 8];

    ax = gca;
    set(ax, 'FontSize', 7, 'LineWidth', 0.5, 'FontName', 'Times New Roman', ...
        'YMinorTick', 'off', 'YTick', 10.^(-5:0));
    xticks([0 200 400 600 800 1000]);
    xlim([0 1000]);
    ylim([1e-5 1e0]);
    box on;

    pdf_path = fullfile(P.out_dir, 'modelA_modelAV_loss.pdf');
    svg_path = fullfile(P.out_dir, 'modelA_modelAV_loss.svg');
    exportgraphics(fig, pdf_path, 'ContentType', 'vector');
    saveas(fig, svg_path);
    fprintf('Saved: %s\n', pdf_path);
end
