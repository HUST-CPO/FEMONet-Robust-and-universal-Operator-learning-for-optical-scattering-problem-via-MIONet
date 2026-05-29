clc
clear all

%model
% Geometry parameters
l1=150:30:300;
%l1=200;
cx=((0:0.1:1)-0.5)*1000;
cy=((0:0.1:1)-0.5)*1000;
theta=0;
% theta=0:pi/4:1.75*pi;

nbr1=length(l1);
nbr2=length(cx);
nbr3=length(cy);
nbr4=length(theta);

script_dir = fileparts(mfilename('fullpath'));
model=mphload(fullfile(script_dir,'comsol_project','scatteringA1.mph'));

% mkdir('scaA1_fig');
% mkdir('scaA1_data');

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
l1=180:40:320;
l2=200:50:350;
cx=(rand(4,1)-0.5)*1000;
cy=(rand(4,1)-0.5)*1000;
% theta=0:pi/4:1.75*pi;

nbr1=length(l1);
nbr2=length(l2);
nbr3=length(cx);
nbr4=length(cy);
% nbr5=length(theta);
all_num=nbr1*nbr2*nbr3*nbr4;

model=mphload(fullfile(script_dir,'comsol_project','scatteringA2.mph'));

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
l1=(180:40:320)*2;
l2=(200:75:350)*2;
cx=(rand(4,1)-0.5)*1000;
cy=(rand(4,1)-0.5)*1000;
theta=0:pi/4:1.75*pi;

nbr1=length(l1);
nbr2=length(l2);
nbr3=length(cx);
nbr4=length(cy);
%nbr5=length(theta);

model=mphload(fullfile(script_dir,'comsol_project','scatteringA3.mph'));
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
l1=320:80:600;
l2=350:100:650;
l3=400:100:600;
cx=(rand(2,1)-0.5)*1000;
cy=(rand(2,1)-0.5)*1000;
%theta=0:pi/4:1.75*pi;

nbr1=length(l1);
nbr2=length(l2);
nbr3=length(l3);
nbr4=length(cx);
nbr5=length(cy);
%nbr6=length(theta);

model=mphload(fullfile(script_dir,'comsol_project','scatteringA4.mph'));

for m=1:nbr5
    for i=1:nbr4
        for j=1:nbr3
            for k=1:nbr2
                for l=1:nbr1
                    nn=nn+1;

                    % Update parameters
                    model.param.set('lda',string(1550)+'[nm]');
                    model.param.set('l1',string(l1(l))+'[nm]');
                    model.param.set('l2',string(l2(k))+'[nm]');
                    model.param.set('l3',string(l3(j))+'[nm]');
                    model.param.set('cx',string(cx(i))+'[nm]');
                    model.param.set('cy',string(cy(m))+'[nm]');

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
end
%%
%model
% Geometry parameters
l2=350:100:650;
l1=0.4:0.15:0.7;
cx=(rand(4,1)-0.5)*1000;
cy=(rand(4,1)-0.5)*1000;
%theta=0:pi/4:1.75*pi;

nbr1=length(l1);
nbr2=length(l2);
nbr3=length(cx);
nbr4=length(cy);


model=mphload(fullfile(script_dir,'comsol_project','scatteringA5.mph'));


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

mat_dir = fullfile(fileparts(mfilename('fullpath')), 'mat_data');
if ~exist(mat_dir, 'dir')
    mkdir(mat_dir);
end
save(fullfile(mat_dir, "Train_data_A_1558"),"Matrix","eps_all",'Ebz_all');