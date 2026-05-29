clc
clear all

mat_dir = fullfile(fileparts(mfilename('fullpath')), 'mat_data');
script_dir = fileparts(mfilename('fullpath'));
if ~exist(mat_dir, 'dir'), mkdir(mat_dir); end

% Merge C1 + C2: 1728 + 1728 = 3456 samples
% Samples per sub-model: 8*3*3*3*2*2*2 = 1728

rng(20260407); % Fixed seed for reproducibility

% Parameter grid
theta = 0:pi/4:1.75*pi; % 8
theta1 = rand(2,1) * 360;
theta2 = rand(2,1) * 360;
theta3 = rand(2,1) * 360;
scale1 = rand(3,1) * 0.4 + 0.8;
scale2 = rand(3,1) * 0.4 + 0.8;
scale3 = rand(3,1) * 0.4 + 0.8;

% C1/C2 geometry and mph files
configs(1).name = 'C1';
configs(1).mph  = 'scatteringC1.mph';
configs(1).l1 = 80;  configs(1).l2 = 320; configs(1).l3 = 120; configs(1).l4 = 200;

configs(2).name = 'C2';
configs(2).mph  = 'scatteringC2.mph';
configs(2).l1 = 90;  configs(2).l2 = 200; configs(2).l3 = 320; configs(2).l4 = 80;

samples_per_model = length(theta) * 3 * 3 * 3 * 2 * 2 * 2; % 1728
total_samples = samples_per_model * length(configs);         % 3456

% Preallocate
Matrix = cell(7, total_samples);
eps_all = zeros(total_samples, 128, 128, 2);
Ebz_all = zeros(total_samples, 128, 128, 2);
model_id = zeros(total_samples, 1); % 1->C1, 2->C2

nn = 0;

for mid = 1:length(configs)
    fprintf('Running model %s (%d/%d)\n', configs(mid).name, mid, length(configs));
    model = mphload(fullfile(script_dir,'comsol_project',configs(mid).mph));

    for m = 1:length(theta)
        for k6 = 1:3
            for k5 = 1:3
                for k4 = 1:3
                    for k3 = 1:2
                        for k2 = 1:2
                            for k1 = 1:2
                                nn = nn + 1;

                                % Update parameters
                                model.param.set('lda', string(1550) + '[nm]');
                                model.param.set('l1', string(configs(mid).l1) + '[nm]');
                                model.param.set('l2', string(configs(mid).l2) + '[nm]');
                                model.param.set('l3', string(configs(mid).l3) + '[nm]');
                                model.param.set('l4', string(configs(mid).l4) + '[nm]');
                                model.param.set('theta1', string(theta1(k1)));
                                model.param.set('theta2', string(theta2(k2)));
                                model.param.set('theta3', string(theta3(k3)));
                                model.param.set('scale1', string(scale1(k4)));
                                model.param.set('scale2', string(scale2(k5)));
                                model.param.set('scale3', string(scale3(k6)));
                                model.param.set('theta', string(theta(m)));

                                % Geometry, mesh, solve
                                geom = model.geom('geom1');
                                geom.run();
                                mesh = model.mesh('mesh1');
                                mesh.run();
                                model.study('std1').run();

                                % Mesh data
                                meshInfo = mphxmeshinfo(model);
                                vertex = meshInfo.nodes.coords';
                                tri = sort(meshInfo.elements.tri.dofs' + 1, 2);

                                % Structured grid interpolation for CNN input
                                n = 128;
                                x_range = linspace(-2, 2, n);
                                y_range = linspace(2, -2, n);
                                [X, Y] = meshgrid(x_range, y_range);
                                points = [X(:), Y(:)] * 1e-6;

                                reps = mphinterp(model, 'real(material.epsilonr_iso)', 'coord', points')';
                                ieps = mphinterp(model, 'imag(material.epsilonr_iso)', 'coord', points')';
                                eps = reshape([reps ieps], n, n, 2);

                                rEbz = mphinterp(model, 'real(ewfd.Ebz)', 'coord', points')';
                                iEbz = mphinterp(model, 'imag(ewfd.Ebz)', 'coord', points')';
                                Ebz = reshape([rEbz iEbz], n, n, 2);

                                % Solution vector and linear system
                                x = mphgetu(model, 'solnum', 1);
                                matrix = mphmatrix(model, 'sol1', 'out', {'K','L'}, 'complexfun', 'on');
                                A = matrix.K;
                                [Ai, Aj, Av] = find(A);

                                Matrix{1, nn} = Ai;
                                Matrix{2, nn} = Aj;
                                Matrix{3, nn} = Av;
                                Matrix{4, nn} = matrix.L;
                                Matrix{5, nn} = x;
                                Matrix{6, nn} = vertex / 1e-6;
                                Matrix{7, nn} = tri;

                                eps_all(nn, :, :, :) = eps;
                                Ebz_all(nn, :, :, :) = Ebz;
                                model_id(nn) = mid;

                                if mod(nn, 50) == 0
                                    fprintf('  Progress: %d / %d\n', nn, total_samples);
                                end
                                close all;
                            end
                        end
                    end
                end
            end
        end
    end
end

fprintf('Finished. Total samples = %d\n', nn);
save(fullfile(mat_dir, 'Train_data_C_3456.mat'), 'Matrix', 'eps_all', 'Ebz_all', 'model_id', '-v7.3');
