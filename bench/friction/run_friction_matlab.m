function run_friction_matlab(out_dir, algorithms, names)
%RUN_FRICTION_MATLAB  fmincon on the friction-audit problems with a user's minimal inputs; JSONL records.
%   run_friction_matlab('..\results\s7-friction')            % interior-point and sqp, every problem
%   Only Algorithm and Display are set (plus SpecifyObjectiveGradient where the user supplied a gradient,
%   which is the mistake wrong_gradient audits). Counting is at the model boundary with a one-point cache.
%   Records carry f and the constraint values at x0 so summarize.py can check the MATLAB and Python
%   definitions agree before it judges the returned points with the Python oracle.
here = fileparts(mfilename('fullpath'));
D = jsondecode(fileread(fullfile(here, 'friction_data.json')));
if nargin < 2 || isempty(algorithms), algorithms = {'interior-point', 'sqp'}; end
if ischar(algorithms), algorithms = strsplit(algorithms, ','); end
if nargin < 3 || isempty(names), names = fieldnames(D)'; end
if ischar(names), names = strsplit(names, ','); end
for a = 1:numel(algorithms)
    alg = algorithms{a};
    fid = fopen(fullfile(out_dir, ['fmincon-' alg '.jsonl']), 'w');
    for k = 1:numel(names)
        rec = run_one(names{k}, alg, D);
        fprintf(fid, '%s\n', jsonencode(rec));
        fprintf('%-22s fmincon-%-15s %-6s exitflag=%d f_model=%d wall=%.3fs %s\n', names{k}, alg, rec.outcome, ...
            rec.exitflag, rec.counts.f_model, rec.wall, strtok(rec.message, newline));
    end
    fclose(fid);
end
end

function rec = run_one(name, alg, D)
P = friction_problems(name, D);
rec = struct('schema', 'mincon-friction-record/1', 'problem', name, 'solver', ['fmincon-' alg], ...
    'solver_version', ['MATLAB ' version('-release') ' optim ' ver('optim').Version], 'n', numel(P.x0), ...
    'user_supplied_gradient', P.specify_gradient, 'started', now * 86400);
cnt = struct('f_model', 0, 'c_model', 0, 'f_calls', 0, 'c_calls', 0, 'nonfinite_f', 0, 'nonfinite_c', 0);
cb = 0; fx = []; fv = []; gv = []; cx = []; cv = {};
    function varargout = fcount(x)
        cnt.f_calls = cnt.f_calls + 1;
        if ~isempty(fx) && isequal(x, fx)
            varargout{1} = fv; if nargout > 1, varargout{2} = gv; end
            return
        end
        t = tic;
        if P.specify_gradient
            [v, g] = P.fun(x); gv = g;
        else
            v = P.fun(x); gv = [];
        end
        cb = cb + toc(t);
        cnt.f_model = cnt.f_model + 1;
        if ~isfinite(v), cnt.nonfinite_f = cnt.nonfinite_f + 1; end
        fx = x; fv = v;
        varargout{1} = v; if nargout > 1, varargout{2} = gv; end
    end
    function [c, ceq] = ccount(x)
        cnt.c_calls = cnt.c_calls + 1;
        if ~isempty(cx) && isequal(x, cx), c = cv{1}; ceq = cv{2}; return; end
        t = tic; [c, ceq] = P.nonlcon(x); cb = cb + toc(t);
        cnt.c_model = cnt.c_model + 1;
        if ~all(isfinite([c(:); ceq(:)])), cnt.nonfinite_c = cnt.nonfinite_c + 1; end
        cx = x; cv = {c, ceq};
    end
% equivalence data at x0 (before any counting)
rec.f0 = P.fun(P.x0);
c0 = [];
if ~isempty(P.A), c0 = [c0; P.A * P.x0 - P.b]; end
if ~isempty(P.Aeq), c0 = [c0; P.Aeq * P.x0 - P.beq]; end
if ~isempty(P.nonlcon), [c, ceq] = P.nonlcon(P.x0); c0 = [c0; c(:); ceq(:)]; end
rec.c0 = c0(:)';
opts = optimoptions('fmincon', 'Algorithm', alg, 'Display', 'off');
if P.specify_gradient, opts = optimoptions(opts, 'SpecifyObjectiveGradient', true); end
rec.options = struct('Algorithm', alg, 'Display', 'off', 'SpecifyObjectiveGradient', P.specify_gradient, 'defaults_otherwise', true);
if isempty(P.nonlcon), fcon = []; else, fcon = @ccount; end
rec.x = []; rec.fval = NaN; rec.exitflag = -99; rec.message = ''; rec.iterations = -1; rec.funcCount = -1;
rec.firstorderopt = NaN; rec.constrviolation = NaN; rec.warnings = {};
lastwarn('');
try
    t0 = tic;
    [x, fval, exitflag, output] = fmincon(@fcount, P.x0, P.A, P.b, P.Aeq, P.beq, P.lb, P.ub, fcon, opts);
    rec.wall = toc(t0);
    rec.x = x(:)'; rec.fval = fval; rec.exitflag = exitflag; rec.message = output.message;
    rec.iterations = output.iterations; rec.funcCount = output.funcCount;
    rec.firstorderopt = output.firstorderopt;
    if isfield(output, 'constrviolation'), rec.constrviolation = output.constrviolation; end
    rec.outcome = 'ok'; rec.error = '';
catch err
    rec.wall = toc(t0);
    rec.outcome = 'error'; rec.error = err.message; rec.message = err.message;
end
[wmsg, ~] = lastwarn();
if ~isempty(wmsg), rec.warnings = {wmsg}; end
rec.reported_success = rec.exitflag > 0;
rec.counts = cnt; rec.callback_seconds = cb;
end
