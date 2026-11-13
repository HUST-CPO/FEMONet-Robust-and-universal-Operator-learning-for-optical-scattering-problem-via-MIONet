function plot_fields_paper_matlab
% Paper-style ablation plots, adapted from A-proceeding/outputE_v3.m.
% Original data/results/scripts are read-only. Outputs use a separate folder.
root = fileparts(mfilename('fullpath'));
dataDir = fullfile(fileparts(root), 'A-proceeding');
outDir = fullfile(root, 'field_comparison_matlab');
if ~exist(outDir,'dir'), mkdir(outDir); end
D = load(fullfile(dataDir,'deepOnet_data_A1_all.mat'));
M = load(fullfile(dataDir,'Train_data_A1_all.mat'),'Matrix');
I = load(fullfile(dataDir,'idx_A1_all.mat'),'testIdx');
R = load(fullfile(root,'runs','raw_seed123','E_test_pred.mat'));
P = load(fullfile(root,'runs','ilu_seed123','E_test_pred.mat'));
A = load(fullfile(root,'runs','verify','E_test_pred.mat'));
selection = jsondecode(fileread(fullfile(root,'field_comparison','plot_manifest.json')));
ids = [selection.selected_samples.test_sample_1based];
percentiles = [selection.selected_samples.selected_percentile];
refs = {A.E_pred, P.E_pred};
refNames = {'table2','paired'};
refLabels = {'FEMONet (ILU, Table 2)','FEMONet (ILU, paired)'};
manifest = struct('source_script','A-proceeding/outputE_v3.m', ...
    'colormap','MATLAB jet(256)','residual','FEM minus prediction, signed real/imaginary components', ...
    'selection','Same P10/P50/P90 no-ILU sample-MSE samples as previous figures', ...
    'color_range','Within each sample/component: all fields share min/max; both errors share symmetric max-abs range', ...
    'sample_test_ids',ids,'outputs',{{}});
for r = 1:numel(refs)
    for j = 1:numel(ids)
        i = ids(j); gid = I.testIdx(i); n = D.coord_len_test(i);
        xy = M.Matrix{6,gid}/1e6;
        tri = M.Matrix{7,gid};
        truth = M.Matrix{5,gid}(:);
        raw = R.E_pred(i,1:n).';
        pre = refs{r}(i,1:n).';
        assert(isequal(truth,D.Ez_test(i,1:n).'));
        assert(isequal(xy*1e6,D.X_test(i,1:n,:)) || ...
            max(abs(xy(:)-reshape(squeeze(D.X_test(i,1:n,:))/1e6,[],1)))<1e-15);
        realData = real([truth raw pre truth-raw truth-pre]);
        imagData = imag([truth raw pre truth-raw truth-pre]);
        fig = figure('Visible','off','Color','w','Units','centimeters', ...
            'Position',[2 2 33 15.5],'Renderer','opengl');
        fig.PaperPositionMode='auto';
        tl = tiledlayout(fig,2,5,'TileSpacing','compact','Padding','compact');
        labels = {'FEM','FEMONet (no ILU)',refLabels{r}, ...
            'FEM - no ILU','FEM - ILU'};
        for component = 1:2
            if component==1, values=realData; fieldLabel='Re'; else, values=imagData; fieldLabel='Im'; end
            cfield = [min(values(:,1:3),[],'all'),max(values(:,1:3),[],'all')];
            em = max(abs(values(:,4:5)),[],'all');
            if em==0, em=1; end
            for k = 1:5
                ax=nexttile(tl);
                if k<=3, limits=cfield; else, limits=[-em em]; end
                drawField(ax,xy,tri,values(:,k),limits);
                title(ax,{labels{k},['$\mathrm{',fieldLabel,'}(E_z^s)$']}, ...
                    'Interpreter','latex','FontName','Times New Roman', ...
                    'FontSize',10,'FontWeight','normal');
            end
        end
        title(tl,sprintf('Case 1 | Test sample %d (source %d; P%d)',i,gid,percentiles(j)), ...
            'FontName','Times New Roman','FontSize',13,'FontWeight','normal','Interpreter','none');
        subtitle(tl,sprintf('Complex field MSE: no ILU = %.3e; ILU = %.3e | Fields in V/m', ...
            mean(abs(truth-raw).^2),mean(abs(truth-pre).^2)), ...
            'FontName','Times New Roman','FontSize',10,'Interpreter','none');
        stem=fullfile(outDir,sprintf('sample_%03d_P%d_%s_real_imag',i,percentiles(j),refNames{r}));
        drawnow;
        print(fig,[stem '.png'],'-dpng','-r250');
        print(fig,[stem '.svg'],'-dsvg','-painters');
        close(fig);
        manifest.outputs{end+1} = [stem '.png'];
        fprintf('Saved %s\n',stem);
    end
end
fid=fopen(fullfile(outDir,'plot_manifest.json'),'w');
fprintf(fid,'%s',jsonencode(manifest,'PrettyPrint',true)); fclose(fid);
end

function drawField(ax,node,elem,Ez,cax)
% Same rendering, exponent and tick conventions as outputE_v3.m.
expPow=getExponentFromCaxis(cax); scale=10^expPow;
trisurf(elem,node(:,1),node(:,2),zeros(size(node,1),1),Ez/scale, ...
    'Parent',ax,'EdgeColor','none');
shading(ax,'interp'); view(ax,2); axis(ax,'equal'); axis(ax,'tight'); axis(ax,'off');
colormap(ax,jet(256)); caxis(ax,cax/scale);
set(ax,'FontName','Times New Roman','LineWidth',0.8);
cb=colorbar(ax,'eastoutside');
cb.FontName='Times New Roman'; cb.FontSize=8;
cb.TickLabelInterpreter='latex'; cb.LineWidth=0.25; cb.Box='on';
[ticks,ticklabels]=getOneSigTicks(cax/scale);
cb.Ticks=ticks; cb.TickLabels=ticklabels;
if expPow~=0
    title(cb,['$\times 10^{' num2str(expPow) '}$'], ...
        'Interpreter','latex','FontName','Times New Roman','FontSize',8);
end
end

function expPow=getExponentFromCaxis(cax)
vmax=max(abs(cax(:)));
if vmax==0, expPow=0; return; end
expPow=floor(log10(vmax));
if abs(expPow)<=1, expPow=0; end
end

function [ticks,ticklabels]=getOneSigTicks(cax_plot)
a=cax_plot(1); b=cax_plot(2);
if a==b, ticks=a; ticklabels={formatOneSig(a)}; return; end
rawStep=(b-a)/4;
p=floor(log10(abs(rawStep))); base=10^p;
candidates=[1 2 4 5 10]*base;
[~,idx]=min(abs(candidates-rawStep)); step=candidates(idx);
ticks=ceil(a/step)*step:step:floor(b/step)*step;
if a<0 && b>0 && ~any(abs(ticks)<1e-12), ticks=sort([ticks 0]); end
ticks(abs(ticks)<1e-12)=0;
ticks=unique(round(ticks,12),'stable');
ticklabels=arrayfun(@formatOneSig,ticks,'UniformOutput',false);
end

function s=formatOneSig(x)
if abs(x)<1e-12, s='0'; else, s=sprintf('%.1g',x); end
end
