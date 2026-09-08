function worker_matlab(solver, track, problems, out_file, progress_file, maxtime, maxfev, experiment, repeat, threads)
%WORKER_MATLAB  Run corpus problems through fmincon; append schema-v1 JSONL records.
%   worker_matlab('fmincon-interior-point', 'A', 'HS71,HS100', 'out.jsonl', 'progress.txt', 60, 100000, 'dev', 0)
%   solver: fmincon-interior-point | fmincon-sqp | fmincon-active-set
%   track A: defaults, no derivatives.  track C: exact objective gradient and constraint Jacobian.
%   Counting happens at the model boundary with a one-point cache (same rule as the Python worker).
%   Only Display and Algorithm are set in track A; the OutputFcn deadline is cooperative — the
%   supervisor enforces the hard limit externally.
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, '..', 'corpus', 'matlab'));
if ischar(maxtime), maxtime = str2double(maxtime); end
if ischar(maxfev), maxfev = str2double(maxfev); end
if ischar(repeat), repeat = str2double(repeat); end
if nargin < 10, threads = 1; end
if ischar(threads), threads = str2double(threads); end
if threads > 0, maxNumCompThreads(threads); end
names = strsplit(problems, ',');
names = names(~cellfun(@isempty, names));
manifest = jsondecode(fileread(fullfile(here, '..', 'corpus', 'manifest.json')));
splitOf = containers.Map();
for i = 1:numel(manifest.problems)
    pr = manifest.problems(i);
    if iscell(pr), pr = pr{1}; end
    if isfield(pr, 'split') && ~isempty(pr.split), splitOf(pr.name) = pr.split; else, splitOf(pr.name) = ''; end
end
algorithm = strrep(solver, 'fmincon-', '');
fid = fopen(out_file, 'a');
for k = 1:numel(names)
    nm = names{k};
    if ~isempty(progress_file)
        pf = fopen(progress_file, 'w'); fprintf(pf, '%s %f\n', nm, now*86400); fclose(pf);
    end
    rec = run_one(nm, solver, algorithm, track, maxtime, maxfev, experiment, repeat, splitOf);
    fprintf(fid, '%s\n', jsonencode(rec));
    fprintf('%-22s %-24s %-6s status=%d f_model=%d c_model=%d wall=%.3fs\n', nm, solver, rec.outcome, rec.native_status, ...
        rec.counts.f_model, rec.counts.c_model, rec.time.solve_wall);
end
fclose(fid);
if ~isempty(progress_file)
    pf = fopen(progress_file, 'w'); fprintf(pf, 'DONE %f\n', now*86400); fclose(pf);
end
end

function rec = run_one(nm, solver, algorithm, track, maxtime, maxfev, experiment, repeat, splitOf)
p = feval(nm);
n = p.n; m = p.m;
rec = struct();
rec.schema = 'mincon-bench-record/1';
rec.run_id = strrep(char(java.util.UUID.randomUUID()), '-', '');
rec.experiment = experiment; rec.track = track; rec.problem = nm; rec.family = p.family;
rec.split = splitOf(nm); rec.checksum = p.checksum; rec.n = n; rec.m = m;
rec.solver = solver; rec.solver_version = ['MATLAB ' version('-release') ' optim ' ver('optim').Version];
rec.derivatives = 'fd'; if strcmp(track, 'C'), rec.derivatives = 'exact'; end
rec.host = struct('platform', computer, 'matlab', version, 'maxNumCompThreads', maxNumCompThreads);
rec.threads = maxNumCompThreads; rec.budget = struct('maxtime', maxtime, 'maxfev', maxfev); rec.repeat = repeat;
rec.notes = {};
% --- counting model with one-point cache ---
cnt = struct('f_model', 0, 'c_model', 0, 'g_model', 0, 'j_model', 0, 'f_calls', 0, 'c_calls', 0, 'g_calls', 0, 'j_calls', 0, 'failed', 0);
cb_seconds = 0;
fx = []; fv = []; cx = []; cv = []; gx = []; gv = []; jx = []; jv = [];
    function v = fcount(x)
        cnt.f_calls = cnt.f_calls + 1;
        if ~isempty(fx) && isequal(x, fx), v = fv; return; end
        t = tic; v = p.f(x); cb_seconds = cb_seconds + toc(t);
        cnt.f_model = cnt.f_model + 1; fx = x; fv = v;
    end
    function g = gcount(x)
        cnt.g_calls = cnt.g_calls + 1;
        if ~isempty(gx) && isequal(x, gx), g = gv; return; end
        t = tic; g = p.grad(x); cb_seconds = cb_seconds + toc(t);
        cnt.g_model = cnt.g_model + 1; gx = x; gv = g;
    end
    function c = ccount(x)
        cnt.c_calls = cnt.c_calls + 1;
        if ~isempty(cx) && isequal(x, cx), c = cv; return; end
        t = tic; c = p.c(x); cb_seconds = cb_seconds + toc(t);
        cnt.c_model = cnt.c_model + 1; cx = x; cv = c;
    end
    function J = jcount(x)
        cnt.j_calls = cnt.j_calls + 1;
        if ~isempty(jx) && isequal(x, jx), J = jv; return; end
        t = tic; J = p.jac(x); cb_seconds = cb_seconds + toc(t);
        cnt.j_model = cnt.j_model + 1; jx = x; jv = J;
    end
% --- canonical -> fmincon translation (exact bound comparison) ---
isEq = (p.cl == p.cu);
loRows = find(isfinite(p.cl) & ~isEq); hiRows = find(isfinite(p.cu) & ~isEq); eqRows = find(isEq);
rec.row_map = struct('hi', hiRows(:)', 'lo', loRows(:)', 'eq', eqRows(:)');
    function [c, ceq] = nonlcon(x)
        cx_ = ccount(x);
        c = [cx_(hiRows) - p.cu(hiRows); p.cl(loRows) - cx_(loRows)];
        ceq = cx_(eqRows) - p.cl(eqRows);
    end
    function [c, ceq, gc, gceq] = nonlcon_grad(x)
        [c, ceq] = nonlcon(x);
        J = jcount(x);
        gc = [J(hiRows, :); -J(loRows, :)]';   % fmincon wants n-by-(#ineq)
        gceq = J(eqRows, :)';
    end
    function [f, g] = obj_grad(x)
        f = fcount(x); g = gcount(x);
    end
lb = p.xl; ub = p.xu;
if strcmp(track, 'C')
    opts = optimoptions('fmincon', 'Algorithm', algorithm, 'Display', 'off', ...
        'SpecifyObjectiveGradient', true, 'SpecifyConstraintGradient', true);
    fobj = @obj_grad; if m > 0, fcon = @nonlcon_grad; else, fcon = []; end
else
    opts = optimoptions('fmincon', 'Algorithm', algorithm, 'Display', 'off');
    fobj = @fcount; if m > 0, fcon = @nonlcon; else, fcon = []; end
end
% cooperative deadline (the supervisor kills the process on the hard limit)
deadline_t0 = tic;
opts = optimoptions(opts, 'OutputFcn', @(x, ov, state) toc(deadline_t0) > maxtime);
rec.options = struct('Algorithm', algorithm, 'Display', 'off', 'defaults_otherwise', true, 'OutputFcn_deadline', maxtime);
rec.x = []; rec.lam = []; rec.zl = []; rec.zu = []; rec.native_status = -99; rec.native_message = ''; rec.reported_success = false;
rec.time = struct('solve_wall', NaN, 'callback', NaN, 'build', 0);
try
    t0 = tic;
    [x, ~, exitflag, output, lambda] = fmincon(fobj, p.x0, [], [], [], [], lb, ub, fcon, opts);
    wall = toc(t0);
    rec.x = x(:)';
    % canonical multipliers: upper rows +ineq, lower rows -ineq, equality rows eqnonlin (fmincon Lagrangian sign)
    lam = zeros(m, 1);
    nh = numel(hiRows);
    if m > 0
        lam(hiRows) = lam(hiRows) + lambda.ineqnonlin(1:nh);
        lam(loRows) = lam(loRows) - lambda.ineqnonlin(nh+1:end);
        lam(eqRows) = lambda.eqnonlin;
    end
    rec.lam = lam(:)'; rec.zl = lambda.lower(:)'; rec.zu = lambda.upper(:)';
    rec.native_status = exitflag; rec.native_message = output.message; rec.reported_success = exitflag > 0;
    rec.nit = output.iterations; rec.funcCount_native = output.funcCount;
    rec.firstorderopt_native = output.firstorderopt; rec.constrviolation_native = output.constrviolation;
    rec.time.solve_wall = wall; rec.time.callback = cb_seconds;
    rec.outcome = 'ok'; rec.error = '';
catch err
    rec.outcome = 'error'; rec.error = err.message; rec.time.callback = cb_seconds;
    rec.native_message = err.message;
end
rec.counts = cnt;
end
