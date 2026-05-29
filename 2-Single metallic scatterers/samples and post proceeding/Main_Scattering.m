clc
clear all

mat_dir = fullfile(fileparts(mfilename('fullpath')), 'mat_data');
script_dir = fileparts(mfilename('fullpath'));
if ~exist(mat_dir, 'dir'), mkdir(mat_dir); end

%l3=200;
%l2=150;
lda=1500:100:2000;
%l1=90;
%l4=80;
%cx=0;
%cy=0;
l3=160:75:310;
l2=100:50:200;
l1=90;
l4=80;
cx=(rand(4,1)-0.5)*1000;
cy=(rand(4,1)-0.5)*1000;
theta=0:pi/4:1.75*pi;
%theta=0:pi/4:pi/4;
%theta=0;

nbr1=length(l3);
nbr2=length(l2);
nbr3=length(cx);
nbr4=length(cy);
nbr5=length(theta);
nbr6=length(lda);
all_num=nbr1*nbr2*nbr3*nbr4*nbr5*nbr6;

model=mphload(fullfile(script_dir,'comsol_project','scatteringB1.mph'));

nn=0;
for o=1:nbr6
    for m=1:nbr5
        for i=1:nbr4
            for j=1:nbr3
                for k=1:nbr2
                    for l=1:nbr1
                        nn=nn+1;

                        % Update parameters
                        model.param.set('lda',string(lda(o))+'[nm]');
                        model.param.set('l3',string(l3(l))+'[nm]');
                        model.param.set('l2',string(l2(k))+'[nm]');
                        model.param.set('l1',string(l1)+'[nm]');
                        model.param.set('l4',string(l4)+'[nm]');
                        model.param.set('cx',string(cx(j))+'[nm]');
                        model.param.set('cy',string(cy(i))+'[nm]');
                        model.param.set('theta',string(theta(m)));

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
                        reps=mphinterp(model, 'real(material.epsilonr_iso)', 'coord', points')';
                        ieps=mphinterp(model, 'imag(material.epsilonr_iso)', 'coord', points')';
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
                        lda_all(nn,1)=lda(o)/1000;
                        eps_all(nn,:,:,:)=eps;
                        Ebz_all(nn,:,:,:)=Ebz;
                        close all;
                    end
                end
            end
        end
    end
end
%%
lda=1500:100:2000;
l3=160:75:310;
l2=100:50:200;
l1=90;
l4=80;
cx=(rand(4,1)-0.5)*1000;
cy=(rand(4,1)-0.5)*1000;
theta=0:pi/4:1.75*pi;

nbr1=length(l3);
nbr2=length(l2);
nbr3=length(cx);
nbr4=length(cy);
nbr5=length(theta);
nbr6=length(lda);

model=mphload(fullfile(script_dir,'comsol_project','scatteringB2.mph'));


for o=1:nbr6
    for m=1:nbr5
        for i=1:nbr4
            for j=1:nbr3
                for k=1:nbr2
                    for l=1:nbr1
                        nn=nn+1;

                        % Update parameters
                        model.param.set('lda',string(lda(o))+'[nm]');
                        model.param.set('l3',string(l3(l))+'[nm]');
                        model.param.set('l2',string(l2(k))+'[nm]');
                        model.param.set('l1',string(l1)+'[nm]');
                        model.param.set('l4',string(l4)+'[nm]');
                        model.param.set('cx',string(cx(j))+'[nm]');
                        model.param.set('cy',string(cy(i))+'[nm]');
                        model.param.set('theta',string(theta(m)));

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
                        reps=mphinterp(model, 'real(material.epsilonr_iso)', 'coord', points')';
                        ieps=mphinterp(model, 'imag(material.epsilonr_iso)', 'coord', points')';
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
                        lda_all(nn,1)=lda(o)/1000;
                        eps_all(nn,:,:,:)=eps;
                        Ebz_all(nn,:,:,:)=Ebz;
                        close all;
                    end
                end
            end
        end
    end
end

%%
lda=1500:100:2000;
l2=160:50:310;
l1=60:20:100;
cx=(rand(4,1)-0.5)*1000;
cy=(rand(4,1)-0.5)*1000;
theta=0:pi/4:1.75*pi;

nbr1=length(l2);
nbr2=length(l1);
nbr3=length(cx);
nbr4=length(cy);
nbr5=length(theta);
nbr6=length(lda);

model=mphload(fullfile(script_dir,'comsol_project','scatteringB3.mph'));

for o=1:nbr6
    for m=1:nbr5
        for i=1:nbr4
            for j=1:nbr3
                for k=1:nbr2
                    for l=1:nbr1
                        nn=nn+1;

                        % Update parameters
                        model.param.set('lda',string(lda(o))+'[nm]');
                        model.param.set('l2',string(l2(l))+'[nm]');
                        model.param.set('l1',string(l1(k))+'[nm]');
                        model.param.set('cx',string(cx(j))+'[nm]');
                        model.param.set('cy',string(cy(i))+'[nm]');
                        model.param.set('theta',string(theta(m)));

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
                        reps=mphinterp(model, 'real(material.epsilonr_iso)', 'coord', points')';
                        ieps=mphinterp(model, 'imag(material.epsilonr_iso)', 'coord', points')';
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
                        lda_all(nn,1)=lda(o)/1000;
                        eps_all(nn,:,:,:)=eps;
                        Ebz_all(nn,:,:,:)=Ebz;
                        close all;
                    end
                end
            end
        end
    end
end
%%
lda=1500:100:2000;
l2=160:75:310;
l4=100:50:200;
l3=120;
l1=80;
cx=(rand(4,1)-0.5)*1000;
cy=(rand(4,1)-0.5)*1000;
theta=0:pi/4:1.75*pi;

nbr1=length(l2);
nbr2=length(l4);
nbr3=length(cx);
nbr4=length(cy);
nbr5=length(theta);
nbr6=length(lda);

model=mphload(fullfile(script_dir,'comsol_project','scatteringB4.mph'));

for o = 1:nbr6
    for m=1:nbr5
        for i=1:nbr4
            for j=1:nbr3
                for k=1:nbr2
                    for l=1:nbr1
                        nn=nn+1;

                        % Update parameters
                        model.param.set('lda',string(lda(o))+'[nm]');
                        model.param.set('l2',string(l2(l))+'[nm]');
                        model.param.set('l4',string(l4(k))+'[nm]');
                        model.param.set('l3',string(l3)+'[nm]');
                        model.param.set('l1',string(l1)+'[nm]');
                        model.param.set('cx',string(cx(j))+'[nm]');
                        model.param.set('cy',string(cy(i))+'[nm]');
                        model.param.set('theta',string(theta(m)));

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
                        reps=mphinterp(model, 'real(material.epsilonr_iso)', 'coord', points')';
                        ieps=mphinterp(model, 'imag(material.epsilonr_iso)', 'coord', points')';
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
                        lda_all(nn,1)=lda(o)/1000;
                        eps_all(nn,:,:,:)=eps;
                        Ebz_all(nn,:,:,:)=Ebz;
                        close all;
                    end
                end
            end
        end
    end
end
%%
lda=1500:100:2000;
l2=160:75:310;
l4=100:50:200;
l3=120;
l1=80;
cx=(rand(4,1)-0.5)*1000;
cy=(rand(4,1)-0.5)*1000;
theta=0:pi/4:1.75*pi;

nbr1=length(l2);
nbr2=length(l4);
nbr3=length(cx);
nbr4=length(cy);
nbr5=length(theta);
nbr6=length(lda);

model=mphload(fullfile(script_dir,'comsol_project','scatteringB5.mph'));

for o=1:nbr6
    for m=1:nbr5
        for i=1:nbr4
            for j=1:nbr3
                for k=1:nbr2
                    for l=1:nbr1
                        nn=nn+1;

                        % Update parameters
                        model.param.set('lda',string(lda(o))+'[nm]');
                        model.param.set('l2',string(l2(l))+'[nm]');
                        model.param.set('l4',string(l4(k))+'[nm]');
                        model.param.set('l3',string(l3)+'[nm]');
                        model.param.set('l1',string(l1)+'[nm]');
                        model.param.set('cx',string(cx(j))+'[nm]');
                        model.param.set('cy',string(cy(i))+'[nm]');
                        model.param.set('theta',string(theta(m)));

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
                        reps=mphinterp(model, 'real(material.epsilonr_iso)', 'coord', points')';
                        ieps=mphinterp(model, 'imag(material.epsilonr_iso)', 'coord', points')';
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
                        lda_all(nn,1)=lda(o)/1000;
                        eps_all(nn,:,:,:)=eps;
                        Ebz_all(nn,:,:,:)=Ebz;
                        close all;
                    end
                end
            end
        end
    end
end
%%
%model l1=90nm l2=156nm l3=210nm l4=83nm
% Geometry parameters

lda=1500:100:2000;
l3=160:75:310;
l2=100:50:200;
l1=90;
l4=80;
cx=(rand(4,1)-0.5)*1000;
cy=(rand(4,1)-0.5)*1000;
theta=0:pi/4:1.75*pi;

nbr1=length(l3);
nbr2=length(l2);
nbr3=length(cx);
nbr4=length(cy);
nbr5=length(theta);
nbr6=length(lda);


model=mphload(fullfile(script_dir,'comsol_project','scatteringB6.mph'));

for o=1:nbr6
    for m=1:nbr5
        for i=1:nbr4
            for j=1:nbr3
                for k=1:nbr2
                    for l=1:nbr1
                        nn=nn+1;

                        % Update parameters
                        model.param.set('lda',string(lda(o))+'[nm]');
                        model.param.set('l3',string(l3(l))+'[nm]');
                        model.param.set('l2',string(l2(k))+'[nm]');
                        model.param.set('l1',string(l1)+'[nm]');
                        model.param.set('l4',string(l4)+'[nm]');
                        model.param.set('cx',string(cy(j))+'[nm]');
                        model.param.set('cy',string(cx(i))+'[nm]');
                        model.param.set('theta',string(theta(m)));

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
                        reps=mphinterp(model, 'real(material.epsilonr_iso)', 'coord', points')';
                        ieps=mphinterp(model, 'imag(material.epsilonr_iso)', 'coord', points')';
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
                        lda_all(nn,1)=lda(o)/1000;
                        eps_all(nn,:,:,:)=eps;
                        Ebz_all(nn,:,:,:)=Ebz;
                        close all;
                    end
                end
            end
        end
    end
end
%%
%model l1=90nm l2=156nm l3=210nm l4=83nm
% Geometry parameters

lda=1500:100:2000;
l2=160:75:310;
l4=100:50:200;
l3=120;
l1=80;
cx=(rand(4,1)-0.5)*1000;
cy=(rand(4,1)-0.5)*1000;
theta=0:pi/4:1.75*pi;

nbr1=length(l2);
nbr2=length(l4);
nbr3=length(cx);
nbr4=length(cy);
nbr5=length(theta);
nbr6=length(lda);

model=mphload(fullfile(script_dir,'comsol_project','scatteringB7.mph'));

for o=1:nbr6
    for m=1:nbr5
        for i=1:nbr4
            for j=1:nbr3
                for k=1:nbr2
                    for l=1:nbr1
                        nn=nn+1;

                        % Update parameters
                        model.param.set('lda',string(lda(o))+'[nm]');
                        model.param.set('l2',string(l2(l))+'[nm]');
                        model.param.set('l4',string(l4(k))+'[nm]');
                        model.param.set('l3',string(l3)+'[nm]');
                        model.param.set('l1',string(l1)+'[nm]');
                        model.param.set('cx',string(cx(j))+'[nm]');
                        model.param.set('cy',string(cy(i))+'[nm]');
                        model.param.set('theta',string(theta(m)));

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
                        reps=mphinterp(model, 'real(material.epsilonr_iso)', 'coord', points')';
                        ieps=mphinterp(model, 'imag(material.epsilonr_iso)', 'coord', points')';
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
                        lda_all(nn,1)=lda(o)/1000;
                        eps_all(nn,:,:,:)=eps;
                        Ebz_all(nn,:,:,:)=Ebz;
                        close all;
                    end
                end
            end
        end
    end
end
%%


save(fullfile(mat_dir, "Train_data_B1_all"),"Matrix","eps_all",'Ebz_all','lda_all');