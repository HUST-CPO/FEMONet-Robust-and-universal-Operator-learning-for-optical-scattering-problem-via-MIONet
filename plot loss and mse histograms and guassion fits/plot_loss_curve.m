function plot_loss_curve(logfile, out_basename, log_format, y_limits)
%PLOT_LOSS_CURVE  Plot train/test loss vs epoch and export PDF + SVG.
%   log_format: 'epoch' | 'columns'
%   y_limits: optional [ymin ymax] for semilogy y-axis

    if nargin < 3 || isempty(log_format)
        log_format = 'epoch';
    end
    if nargin < 4
        y_limits = [];
    end

    P = get_plot_paths();
    if ~isfile(logfile)
        error('Log file not found: %s', logfile);
    end

    txt = fileread(logfile);
    numPat = '[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?';

    switch lower(log_format)
        case 'epoch'
            pattern = ['Epoch\s+(\d+),\s+Total Loss:\s*(' numPat '),\s*test Loss\s*(' numPat ')'];
            tokens = regexp(txt, pattern, 'tokens');
        case 'columns'
            pattern = ['^\s*(\d+)\s+(' numPat ')\s+(' numPat ')\s+(' numPat ')\s*$'];
            tokens = regexp(txt, pattern, 'tokens', 'lineanchors');
        otherwise
            error('Unknown log_format: %s', log_format);
    end

    n = numel(tokens);
    if n == 0
        error('No loss records parsed from: %s', logfile);
    end

    epochs = zeros(n, 1);
    train_loss = zeros(n, 1);
    test_loss = zeros(n, 1);
    for k = 1:n
        epochs(k) = str2double(tokens{k}{1});
        train_loss(k) = str2double(tokens{k}{2});
        test_loss(k) = str2double(tokens{k}{3});
    end

    fig = figure('Color', 'w', 'Units', 'centimeters', 'Position', [10 10 5 4]);
    semilogy(epochs, train_loss, 'b-', 'LineWidth', 0.5); hold on;
    semilogy(epochs, test_loss, 'r-', 'LineWidth', 0.5);

    xlabel('Epoch');
    ylabel('Loss Function');
    lgd = legend('Train', 'Test', 'Location', 'northeast', 'Box', 'off');
    lgd.FontName = 'Times New Roman';
    lgd.ItemTokenSize = [8, 8];

    ax = gca;
    set(ax, 'FontSize', 7, 'LineWidth', 0.5, 'FontName', 'Times New Roman', ...
        'YMinorTick', 'off');
    xticks([0 200 400 600 800 1000]);
    xlim([0 1000]);
    box on;

    if ~isempty(y_limits)
        ylim(y_limits);
    elseif strcmp(log_format, 'columns')
        ax.YTick = 10.^(-6:0);
    else
        ax.YTick = 10.^(-3:2);
        ylim([1e-3 1e2]);
    end

    pdf_path = fullfile(P.out_dir, [out_basename, '.pdf']);
    svg_path = fullfile(P.out_dir, [out_basename, '.svg']);
    exportgraphics(fig, pdf_path, 'ContentType', 'vector');
    saveas(fig, svg_path);
    fprintf('Saved loss curve: %s\n', pdf_path);
end
