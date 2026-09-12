function P = friction_problems(name, D)
%FRICTION_PROBLEMS  The friction-audit problems as a MATLAB user would write them for fmincon.
%   P = friction_problems(name, D) with D = jsondecode(fileread('friction_data.json')).
%   Fields: fun, x0, A, b, Aeq, beq, lb, ub, nonlcon ([] when absent), specify_gradient (wrong_gradient only).
%   The definitions mirror bench/friction/problems.py term by term; the equivalence at x0 is checked by
%   summarize.py against the f0/c0 values exported with the data.
d = D.(name);
P = struct('name', name, 'x0', d.x0(:), 'A', [], 'b', [], 'Aeq', [], 'beq', [], 'lb', [], 'ub', [], ...
           'nonlcon', [], 'specify_gradient', false);
if isfield(d, 'A') && ~isempty(d.A), P.A = reshape_matrix(d.A, numel(d.b), numel(d.x0)); P.b = d.b(:); end
if isfield(d, 'Aeq') && ~isempty(d.Aeq), P.Aeq = reshape_matrix(d.Aeq, numel(d.beq), numel(d.x0)); P.beq = d.beq(:); end
P.lb = bound_vector(d.lb, -Inf); P.ub = bound_vector(d.ub, Inf);
switch name
    case 'chainrosen20'
        P.fun = @(x) sum(100*(x(2:end) - x(1:end-1).^2).^2 + (1 - x(1:end-1)).^2);
    case 'odefit'
        t = d.t(:); data = d.data(:);
        P.fun = @(x) odefit_obj(x, t, data);
    case 'portfolio_risk'
        Sigma = d.Sigma; mu = d.mu(:); risk = d.risk;
        P.fun = @(x) -mu' * x;
        P.nonlcon = @(x) deal(x' * Sigma * x - risk, []);
    case 'pressure_vessel'
        P.fun = @(x) 0.6224*x(1)*x(3)*x(4) + 1.7781*x(2)*x(3)^2 + 3.1661*x(1)^2*x(4) + 19.84*x(1)^2*x(3);
        P.nonlcon = @(x) deal([-x(1) + 0.0193*x(3); -x(2) + 0.00954*x(3); ...
                               -pi*x(3)^2*x(4) - 4/3*pi*x(3)^3 + 1296000; x(4) - 240], []);
    case 'nan_region'
        P.fun = @(x) nan_region_obj(x);
        P.nonlcon = @(x) deal(x(1)^2 + x(2)^2 - 1, []);
    case 'hs71'
        P.fun = @(x) x(1)*x(4)*(x(1) + x(2) + x(3)) + x(3);
        P.nonlcon = @(x) deal(25 - x(1)*x(2)*x(3)*x(4), x(1)^2 + x(2)^2 + x(3)^2 + x(4)^2 - 40);
    case 'linear_only'
        Q = d.Q; c = d.c(:);
        P.fun = @(x) 0.5 * x' * Q * x - c' * x;
    case 'with_args'
        t = d.t(:); dd = d.d(:);
        model = @(x, tt) x(1) * exp(-x(2) * tt) + x(3);
        P.fun = @(x) sum((model(x, t) - dd).^2);       % the data are captured by the handle
    case 'wrong_gradient'
        tgt = d.tgt(:);
        P.fun = @(x) wrong_gradient_obj(x, tgt);
        P.specify_gradient = true;
    case 'infeasible_start_far'
        c = d.c(:);
        P.fun = @(x) 0.5 * sum((x - c).^2);
    case 'bad_scaling'
        P.fun = @(x) (x(1)/3e6 - 1)^2 + (x(2)/1e-6 - 1)^2;
        P.nonlcon = @(x) deal(5 - x(1)*x(2), []);
    case 'noisy_simulator'
        tgt = d.tgt(:); w = d.w(:); primes = d.primes(:); amp = d.amp;
        P.fun = @(x) sum(w .* (x - tgt).^2) + amp * sin(1e5 * (primes' * x));
    case 'box_lsq'
        K = d.K; y = d.y(:);
        P.fun = @(x) 0.5 * sum((K * x - y).^2);
    case 'equality_circle'
        P.fun = @(x) -x(3) + 0.1 * (x(1)^2 + x(2)^2);
        P.nonlcon = @(x) deal(x(1) - 0.3, [x(1)^2 + x(2)^2 + x(3)^2 - 1; x(1) + x(2) - 0.5]);
    otherwise
        error('unknown friction problem %s', name);
end
end

function M = reshape_matrix(v, rows, cols)
M = reshape(double(v), rows, cols);
if rows == 1 && cols > 1 && size(v, 1) ~= 1 && size(v, 2) ~= 1, M = reshape(double(v), rows, cols); end
if ~isequal(size(M), [rows cols]), M = double(v); end
end

function v = bound_vector(raw, fill)
% jsondecode turns [null, 0, null] into a cell array; finite-only vectors stay numeric.
if iscell(raw)
    v = zeros(numel(raw), 1);
    for i = 1:numel(raw)
        if isempty(raw{i}), v(i) = fill; else, v(i) = raw{i}; end
    end
else
    v = double(raw(:));
    v(isnan(v)) = fill;    % jsondecode maps null to NaN in numeric arrays
end
end

function f = odefit_obj(x, t, data)
r = x(1); K = x(2); y0 = x(3);
opts = odeset('RelTol', 1e-10, 'AbsTol', 1e-12);
[~, y] = ode45(@(tt, yy) r * yy * (1 - yy / K), [0; t], y0, opts);
f = sum((y(2:end) - data).^2);
end

function f = nan_region_obj(x)
r2 = 1.5 - x(1)^2 - x(2)^2;
if r2 < 0
    f = NaN;
else
    f = (x(1) - 0.9)^2 + (x(2) - 0.9)^2 + 0.1 * sqrt(r2);
end
end

function [f, g] = wrong_gradient_obj(x, tgt)
f = sum((x - tgt).^2) + 0.1 * sum(x.^4);
if nargout > 1
    g = 2 * (x - tgt) + 0.4 * x.^3;
    g(2) = -g(2);   % the user's mistake
end
end
