% Direct MATLAB fmincon runs on five tiny analytic problems, written as JSON.
% Usage: matlab -batch "run('parity_matlab.m')"  (writes parity_matlab.json next to this file)
here = fileparts(mfilename('fullpath'));
P = parity_problems();
out = struct('matlab_release', version('-release'), 'optim_version', ver('optim').Version, 'runs', {{}});
for k = 1:numel(P)
    p = P{k};
    for alg = {'interior-point', 'sqp'}
        opts = optimoptions('fmincon', 'Display', 'off', 'Algorithm', alg{1});
        t0 = tic;
        [x, f, exitflag, output, lambda] = fmincon(p.fun, p.x0, p.A, p.b, p.Aeq, p.beq, p.lb, p.ub, p.nonlcon, opts);
        t = toc(t0);
        r = struct('problem', p.name, 'algorithm', alg{1}, 'x', x(:)', 'fun', f, 'exitflag', exitflag, ...
            'iterations', output.iterations, 'funcCount', output.funcCount, ...
            'constrviolation', output.constrviolation, 'firstorderopt', output.firstorderopt, ...
            'wall_seconds', t, 'lambda', lambda);
        out.runs{end+1} = r;
    end
end
fid = fopen(fullfile(here, 'parity_matlab.json'), 'w'); fprintf(fid, '%s', jsonencode(out)); fclose(fid);
disp('wrote parity_matlab.json');

function P = parity_problems()
P = {};
% P1 linear inequality
P{end+1} = prob('P1_linear_ineq', @(x) sum((x-1).^2), [0;0], [1 1], 1, [], [], [], [], []);
% P2 nonlinear inequality: min x1+x2 s.t. x1^2+x2^2 <= 1
P{end+1} = prob('P2_nonlinear_ineq', @(x) x(1)+x(2), [0.5;0.5], [], [], [], [], [], [], @(x) deal(x(1)^2+x(2)^2-1, []));
% P3 nonlinear equality: min x1^2+x2^2 s.t. x1*x2 = 1
P{end+1} = prob('P3_nonlinear_eq', @(x) x(1)^2+x(2)^2, [2;0.5], [], [], [], [], [], [], @(x) deal([], x(1)*x(2)-1));
% P4 bounds: min (x1-3)^2+(x2+1)^2, 0<=x<=2
P{end+1} = prob('P4_bounds', @(x) (x(1)-3)^2+(x(2)+1)^2, [1;1], [], [], [], [], [0;0], [2;2], []);
% P5 ranged linear: 1 <= x1+x2 <= 2, min (x1-3)^2+(x2-3)^2
P{end+1} = prob('P5_ranged_linear', @(x) (x(1)-3)^2+(x(2)-3)^2, [0;0], [1 1; -1 -1], [2; -1], [], [], [], [], []);
end

function p = prob(name, fun, x0, A, b, Aeq, beq, lb, ub, nonlcon)
p = struct('name', name, 'fun', fun, 'x0', x0, 'A', A, 'b', b, 'Aeq', Aeq, 'beq', beq, 'lb', lb, 'ub', ub, 'nonlcon', nonlcon);
end
