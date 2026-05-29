clc
clear all

%model
% Geometry parameters
l1=150:50:300;
cx=(rand(4,1)-0.5)*1000;
cy=(rand(4,1)-0.5)*1000;
theta=0;

nbr1=length(l1);
nbr2=length(cx);
nbr3=length(cy);
nbr4=length(theta);

script_dir = fileparts(mfilename('fullpath'));
model=mphload(fullfile(script_dir,'comsol_project','scatteringA6.mph'));

nn=0;
for i=1:nbr4
    for j=1:nbr3
        for k=1:nbr2
            for l=1:nbr1
                nn=nn+1;

                % Update parameters
                model.param.set('lda',string(1550)+'[nm]');
                model.param.set('l1',string(l1(l))+'[nm]');
                model.param.set('cx',string(cx(k))+'[nm]');
                model.param.set('cy',string(cy(j))+'[nm]');

                % Update geometry
                geom=model.geom('geom1');
                geom.run();

                % Update mesh
                mesh=model.mesh('mesh1');
                mesh.run();

                % Solve
                model.study('std1').run();
                % Extract mesh info
                meshInfo = mphxmeshinfo(model);
                vertex=meshInfo.nodes.coords';
                tri=sort(meshInfo.elements.tri.dofs'+1,2);
                edge=GetEdge(tri);
                n = 128;  % 64x64 grid
                x_range = linspace(-1, 1, n);
                y_range = linspace(1, -1, n);
                [X, Y] = meshgrid(x_range, y_range);
                points = [X(:), Y(:)]*1e-6;
                reps=mphinterp(model, 'ewfd.n_iso', 'coord', points')';
                ieps=mphinterp(model, 'ewfd.ki_iso', 'coord', points')';
                eps=[reps ieps];
                eps = reshape(eps, 128, 128, 2);

                rEbz=mphinterp(model, 'real(ewfd.Ebz)', 'coord', points')';
                iEbz=mphinterp(model, 'imag(ewfd.Ebz)', 'coord', points')';
                Ebz=[rEbz iEbz];
                Ebz = reshape(Ebz, 128, 128, 2);

                % Results
                x = mphgetu(model,'solnum',1);
                rEsz=real(x);
                iEsz=imag(x);
                Esz=[rEsz iEsz];
                matrix = mphmatrix(model, 'sol1', 'out',{'K','L'},'complexfun','on');

                % Save figures
                % figure('visible','off');
                % PlotE(vertex,tri,rEsz,0);
                % saveas(gcf, 'scaA1_fig\rEsz_scaA1_'+string(nn), 'png');
                % PlotE(vertex,tri,iEsz,0);
                % saveas(gcf, 'scaA1_fig\iEsz_scaA1_'+string(nn), 'png');
                % epsilon=GetEps(vertex,model);
                % PlotE(vertex,tri,epsilon(:,1),1);
                % saveas(gcf, 'scaA1_fig\mesh_scaA1_'+string(nn), 'png');

                % Save data
                % SaveData(vertex,edge,tri,Ebz,Esz,eps,matrix,'scaA1_','scaA1_data\',nn);

                A=matrix.K;
                [Ai,Aj,Av]=find(A);
                Matrix{1,nn}= Ai;
                Matrix{2,nn}= Aj;
                Matrix{3,nn}= Av;
                Matrix{4,nn}= matrix.L;
                Matrix{5,nn}= x;
                Matrix{6,nn}= vertex/1e-6;
                Matrix{7,nn}= tri;
                eps_all(nn,:,:,:)=eps;
                Ebz_all(nn,:,:,:)=Ebz;
                close all;
            end
        end
    end
end
%%
%model
% Geometry parameters
l1= 180:40:320;
l2= 200*2;
cx=(rand(4,1)/2-0.5)*1000;
cy=(rand(4,1)/2-0.5)*1000;
theta=0:pi/4:1.75*pi;

nbr1=length(l1);
nbr2=length(l2);
nbr3=length(cx);
nbr4=length(cy);
%nbr5=length(theta);

model=mphload(fullfile(script_dir,'comsol_project','scatteringA7.mph'));
for i=1:nbr4
    for j=1:nbr3
        for k=1:nbr2
            for l=1:nbr1
                nn=nn+1;

                % Update parameters
                model.param.set('lda',string(1550)+'[nm]');
                model.param.set('l1',string(l1(l))+'[nm]');
                model.param.set('l2',string(l2(k))+'[nm]');
                model.param.set('cx',string(cx(j))+'[nm]');
                model.param.set('cy',string(cy(i))+'[nm]');

                % Update geometry
                geom=model.geom('geom1');
                geom.run();

                % Update mesh
                mesh=model.mesh('mesh1');
                mesh.run();

                % Solve
                model.study('std1').run();
                % Extract mesh info
                meshInfo = mphxmeshinfo(model);
                vertex=meshInfo.nodes.coords';
                tri=sort(meshInfo.elements.tri.dofs'+1,2);
                edge=GetEdge(tri);
                n = 128;  % 64x64 grid
                x_range = linspace(-1, 1, n);
                y_range = linspace(1, -1, n);
                [X, Y] = meshgrid(x_range, y_range);
                points = [X(:), Y(:)]*1e-6;
                reps=mphinterp(model, 'ewfd.n_iso', 'coord', points')';
                ieps=mphinterp(model, 'ewfd.ki_iso', 'coord', points')';
                eps=[reps ieps];
                eps = reshape(eps, 128, 128, 2);

                rEbz=mphinterp(model, 'real(ewfd.Ebz)', 'coord', points')';
                iEbz=mphinterp(model, 'imag(ewfd.Ebz)', 'coord', points')';
                Ebz=[rEbz iEbz];
                Ebz = reshape(Ebz, 128, 128, 2);

                % Results
                x = mphgetu(model,'solnum',1);
                rEsz=real(x);
                iEsz=imag(x);
                Esz=[rEsz iEsz];
                matrix = mphmatrix(model, 'sol1', 'out',{'K','L'},'complexfun','on');

                % Save data
                A=matrix.K;
                [Ai,Aj,Av]=find(A);
                Matrix{1,nn}= Ai;
                Matrix{2,nn}= Aj;
                Matrix{3,nn}= Av;
                Matrix{4,nn}= matrix.L;
                Matrix{5,nn}= x;
                Matrix{6,nn}= vertex/1e-6;
                Matrix{7,nn}= tri;
                eps_all(nn,:,:,:)=eps;
                Ebz_all(nn,:,:,:)=Ebz;
                close all;
            end
        end
    end
end

%%
%model
% Geometry parameters
l2=350:100:650;
l1=0.4;
cx=(rand(4,1)-0.5)*1000;
cy=(rand(4,1)-0.5)*1000;
%theta=0:pi/4:1.75*pi;

nbr1=length(l1);
nbr2=length(l2);
nbr3=length(cx);
nbr4=length(cy);


model=mphload(fullfile(script_dir,'comsol_project','scatteringA8.mph'));


for i=1:nbr4
    for j=1:nbr3
        for k=1:nbr2
            for l=1:nbr1
                nn=nn+1;

                % Update parameters
                model.param.set('lda',string(1550)+'[nm]');
                model.param.set('l1',string(l1(l)*l2(k))+'[nm]');
                model.param.set('l2',string(l2(k))+'[nm]');
                model.param.set('cx',string(cx(j))+'[nm]');
                model.param.set('cy',string(cy(i))+'[nm]');

                % Update geometry
                geom=model.geom('geom1');
                geom.run();

                % Update mesh
                mesh=model.mesh('mesh1');
                mesh.run();

                % Solve
                model.study('std1').run();
                % Extract mesh info
                meshInfo = mphxmeshinfo(model);
                vertex=meshInfo.nodes.coords';
                tri=sort(meshInfo.elements.tri.dofs'+1,2);
                edge=GetEdge(tri);
                n = 128;  % 64x64 grid
                x_range = linspace(-1, 1, n);
                y_range = linspace(1, -1, n);
                [X, Y] = meshgrid(x_range, y_range);
                points = [X(:), Y(:)]*1e-6;
                reps=mphinterp(model, 'ewfd.n_iso', 'coord', points')';
                ieps=mphinterp(model, 'ewfd.ki_iso', 'coord', points')';
                eps=[reps ieps];
                eps = reshape(eps, 128, 128, 2);

                rEbz=mphinterp(model, 'real(ewfd.Ebz)', 'coord', points')';
                iEbz=mphinterp(model, 'imag(ewfd.Ebz)', 'coord', points')';
                Ebz=[rEbz iEbz];
                Ebz = reshape(Ebz, 128, 128, 2);

                % Results
                x = mphgetu(model,'solnum',1);
                rEsz=real(x);
                iEsz=imag(x);
                Esz=[rEsz iEsz];
                matrix = mphmatrix(model, 'sol1', 'out',{'K','L'},'complexfun','on');

                % Save data
                A=matrix.K;
                [Ai,Aj,Av]=find(A);
                Matrix{1,nn}= Ai;
                Matrix{2,nn}= Aj;
                Matrix{3,nn}= Av;
                Matrix{4,nn}= matrix.L;
                Matrix{5,nn}= x;
                Matrix{6,nn}= vertex/1e-6;
                Matrix{7,nn}= tri;
                eps_all(nn,:,:,:)=eps;
                Ebz_all(nn,:,:,:)=Ebz;
                close all;
            end
        end
    end
end
%% 
%model
% Geometry parameters
l1=150:50:300;
cx=(rand(4,1)-0.5)*1000;
cy=(rand(4,1)-0.5)*1000;
theta=0;

nbr1=length(l1);
nbr2=length(cx);
nbr3=length(cy);
nbr4=length(theta);

model=mphload(fullfile(script_dir,'comsol_project','scatteringA9.mph'));

for i=1:nbr4
    for j=1:nbr3
        for k=1:nbr2
            for l=1:nbr1
                nn=nn+1;

                % Update parameters
                model.param.set('lda',string(1550)+'[nm]');
                model.param.set('l1',string(l1(l))+'[nm]');
                model.param.set('cx',string(cx(k))+'[nm]');
                model.param.set('cy',string(cy(j))+'[nm]');

                % Update geometry
                geom=model.geom('geom1');
                geom.run();

                % Update mesh
                mesh=model.mesh('mesh1');
                mesh.run();

                % Solve
                model.study('std1').run();
                % Extract mesh info
                meshInfo = mphxmeshinfo(model);
                vertex=meshInfo.nodes.coords';
                tri=sort(meshInfo.elements.tri.dofs'+1,2);
                edge=GetEdge(tri);
                n = 128;  % 64x64 grid
                x_range = linspace(-1, 1, n);
                y_range = linspace(1, -1, n);
                [X, Y] = meshgrid(x_range, y_range);
                points = [X(:), Y(:)]*1e-6;
                reps=mphinterp(model, 'ewfd.n_iso', 'coord', points')';
                ieps=mphinterp(model, 'ewfd.ki_iso', 'coord', points')';
                eps=[reps ieps];
                eps = reshape(eps, 128, 128, 2);

                rEbz=mphinterp(model, 'real(ewfd.Ebz)', 'coord', points')';
                iEbz=mphinterp(model, 'imag(ewfd.Ebz)', 'coord', points')';
                Ebz=[rEbz iEbz];
                Ebz = reshape(Ebz, 128, 128, 2);

                % Results
                x = mphgetu(model,'solnum',1);
                rEsz=real(x);
                iEsz=imag(x);
                Esz=[rEsz iEsz];
                matrix = mphmatrix(model, 'sol1', 'out',{'K','L'},'complexfun','on');

                A=matrix.K;
                [Ai,Aj,Av]=find(A);
                Matrix{1,nn}= Ai;
                Matrix{2,nn}= Aj;
                Matrix{3,nn}= Av;
                Matrix{4,nn}= matrix.L;
                Matrix{5,nn}= x;
                Matrix{6,nn}= vertex/1e-6;
                Matrix{7,nn}= tri;
                eps_all(nn,:,:,:)=eps;
                Ebz_all(nn,:,:,:)=Ebz;
                close all;
            end
        end
    end
end
%% 
%model
% Geometry parameters
r1=150:50:300;
cx=(rand(4,1)-0.5)*1000;
cy=(rand(4,1)-0.5)*1000;
theta=0;

nbr1=length(r1);
nbr2=length(cx);
nbr3=length(cy);
nbr4=length(theta);

model=mphload(fullfile(script_dir,'comsol_project','scatteringA10.mph'));

for i=1:nbr4
    for j=1:nbr3
        for k=1:nbr2
            for l=1:nbr1
                nn=nn+1;

                % Update parameters
                model.param.set('lda',string(1550)+'[nm]');
                model.param.set('r1',string(r1(l))+'[nm]');
                model.param.set('cx',string(cx(k))+'[nm]');
                model.param.set('cy',string(cy(j))+'[nm]');

                % Update geometry
                geom=model.geom('geom1');
                geom.run();

                % Update mesh
                mesh=model.mesh('mesh1');
                mesh.run();

                % Solve
                model.study('std1').run();
                % Extract mesh info
                meshInfo = mphxmeshinfo(model);
                vertex=meshInfo.nodes.coords';
                tri=sort(meshInfo.elements.tri.dofs'+1,2);
                edge=GetEdge(tri);
                n = 128;  % 64x64 grid
                x_range = linspace(-1, 1, n);
                y_range = linspace(1, -1, n);
                [X, Y] = meshgrid(x_range, y_range);
                points = [X(:), Y(:)]*1e-6;
                reps=mphinterp(model, 'ewfd.n_iso', 'coord', points')';
                ieps=mphinterp(model, 'ewfd.ki_iso', 'coord', points')';
                eps=[reps ieps];
                eps = reshape(eps, 128, 128, 2);

                rEbz=mphinterp(model, 'real(ewfd.Ebz)', 'coord', points')';
                iEbz=mphinterp(model, 'imag(ewfd.Ebz)', 'coord', points')';
                Ebz=[rEbz iEbz];
                Ebz = reshape(Ebz, 128, 128, 2);

                % Results
                x = mphgetu(model,'solnum',1);
                rEsz=real(x);
                iEsz=imag(x);
                Esz=[rEsz iEsz];
                matrix = mphmatrix(model, 'sol1', 'out',{'K','L'},'complexfun','on');

                A=matrix.K;
                [Ai,Aj,Av]=find(A);
                Matrix{1,nn}= Ai;
                Matrix{2,nn}= Aj;
                Matrix{3,nn}= Av;
                Matrix{4,nn}= matrix.L;
                Matrix{5,nn}= x;
                Matrix{6,nn}= vertex/1e-6;
                Matrix{7,nn}= tri;
                eps_all(nn,:,:,:)=eps;
                Ebz_all(nn,:,:,:)=Ebz;
                close all;
            end
        end
    end
end

mat_dir = fullfile(fileparts(mfilename('fullpath')), 'mat_data');
if ~exist(mat_dir, 'dir')
    mkdir(mat_dir);
end
save(fullfile(mat_dir, "Train_data_A_unseen"),"Matrix","eps_all",'Ebz_all');