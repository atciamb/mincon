function fmincon_baseline(problem_list, out_file, algorithm, maxtime)
%FMINCON_BASELINE  Generate the fmincon reference results for the mincon benchmark.
%
%   fmincon_baseline('problems.txt', 'fmincon_results.jsonl')
%   fmincon_baseline('problems.txt', 'out.jsonl', 'sqp', 60)
%
% WHY THIS FILE EXISTS
%
% fmincon is the solver we are measuring ourselves against and it needs a
% MATLAB licence, so it cannot run in this project's CI. The honest way to
% handle that is to make it trivial for anyone who *does* have a licence to
% produce the baseline and drop it into the comparison, rather than quoting
% numbers from a paper that used a different problem set, a different machine
% and a different MATLAB version.
%
% Run this, commit the resulting .jsonl next to the other results, and
% profiles.py will treat 'fmincon' as one more solver. Its records have exactly
% the same schema as runner.py's.
%
% REQUIREMENTS
%
%   * MATLAB with the Optimization Toolbox.
%   * The S2MPJ MATLAB problems on the path:
%       git clone --depth 1 https://github.com/GrattonToint/S2MPJ
%       addpath('S2MPJ/matlab_problems')
%
% FAIRNESS RULES  -- these are not optional if the comparison is to mean anything
%
%   1. No gradients are supplied, to either solver. The plug-and-play case is
%      what fmincon is famous for and it is the case we claim to beat.
%   2. Default options otherwise, except the time limit. Tuning fmincon down or
%      mincon up invalidates the comparison.
%   3. The success criterion is applied later, by profiles.py, from the
%      returned point. This script records what happened; it does not judge.
%   4. Record the MATLAB version. fmincon changes between releases.

    if nargin < 3 || isempty(algorithm), algorithm = 'interior-point'; end
    if nargin < 4 || isempty(maxtime),   maxtime   = 60; end

    names = readlines(problem_list);
    names = names(strlength(names) > 0);

    fid = fopen(out_file, 'w');
    cleaner = onCleanup(@() fclose(fid));

    meta = struct('kind', 'meta', 'solver', ['fmincon-' algorithm], ...
                  'matlab', version(), 'computer', computer(), ...
                  'date', datestr(now, 31), 'problems', numel(names));
    fprintf(fid, '%s\n', jsonencode(meta));

    for k = 1:numel(names)
        name = strtrim(names(k));
        fprintf('[%d/%d] %s\n', k, numel(names), name);
        rec = run_one(name, algorithm, maxtime);
        fprintf(fid, '%s\n', jsonencode(rec));
    end
    fprintf('wrote %s\n', out_file);
end


function rec = run_one(name, algorithm, maxtime)
    rec = struct('problem', char(name), 'solver', ['fmincon-' algorithm], ...
                 'n', 0, 'm', 0, 'classification', '', 'ok', false, ...
                 'f', inf, 'maxcv', inf, 'nfev', 0, 'nit', 0, 'seconds', 0, ...
                 'reported_success', false, 'reported_status', '', ...
                 'error', '', 'notes', {{}});
    try
        prob = feval(char(name), 'setup');
    catch err
        rec.error = ['setup failed: ' err.message];
        return
    end

    n = double(prob.n);  m = double(prob.m);
    rec.n = n;  rec.m = m;
    if isfield(prob, 'pbclass'), rec.classification = char(prob.pbclass); end

    x0 = full(prob.x0(:));
    lb = full(prob.xlower(:));  ub = full(prob.xupper(:));
    lb(lb <= -1e20) = -inf;     ub(ub >= 1e20) = inf;

    if m > 0
        cl = full(prob.clower(:)); cu = full(prob.cupper(:));
        cl(cl <= -1e20) = -inf;    cu(cu >= 1e20) = inf;
        isEq = (cl == cu);
    else
        cl = []; cu = []; isEq = [];
    end

    % Count evaluations here rather than trusting output.funcCount, so the
    % number is comparable with the one runner.py computes for every other
    % solver.
    counter = struct('n', 0);
    function f = objective(x)
        counter.n = counter.n + 1;
        f = feval(char(name), 'fx', x);
    end

    % fmincon wants c(x) <= 0 and ceq(x) = 0. Translate the canonical
    % two-sided rows into that shape: an equality row becomes ceq, a row with a
    % finite lower bound becomes cl - c <= 0, one with a finite upper bound
    % becomes c - cu <= 0, and a range row contributes both.
    function [c, ceq] = nonlcon(x)
        if m == 0
            c = []; ceq = []; return
        end
        cx = full(feval(char(name), 'cx', x));
        cx = cx(:);
        ceq = cx(isEq) - cl(isEq);
        c = [];
        loRows = find(~isEq & isfinite(cl));
        hiRows = find(~isEq & isfinite(cu));
        if ~isempty(loRows), c = [c; cl(loRows) - cx(loRows)]; end %#ok<AGROW>
        if ~isempty(hiRows), c = [c; cx(hiRows) - cu(hiRows)]; end %#ok<AGROW>
    end

    opts = optimoptions('fmincon', ...
        'Algorithm', algorithm, ...
        'Display', 'off', ...
        'MaxFunctionEvaluations', 100000, ...
        'MaxIterations', 3000, ...
        'OutputFcn', @(x, ov, state) deadline(ov, maxtime));

    t0 = tic;
    try
        [x, ~, exitflag, output] = fmincon(@objective, x0, [], [], [], [], ...
                                           lb, ub, @nonlcon, opts);
        rec.seconds = toc(t0);
        rec.ok = true;
        rec.nfev = counter.n;
        rec.nit = double(output.iterations);
        rec.reported_success = exitflag > 0;
        rec.reported_status = num2str(exitflag);

        % Recompute f and the violation ourselves, on the original problem.
        rec.f = full(feval(char(name), 'fx', x));
        v = max([0; lb(isfinite(lb)) - x(isfinite(lb)); x(isfinite(ub)) - ub(isfinite(ub))]);
        if m > 0
            cx = full(feval(char(name), 'cx', x)); cx = cx(:);
            v = max([v; cl(isfinite(cl)) - cx(isfinite(cl)); cx(isfinite(cu)) - cu(isfinite(cu))]);
        end
        rec.maxcv = max(v, 0);
    catch err
        rec.seconds = toc(t0);
        rec.error = err.message;
        rec.nfev = counter.n;
    end
end


function stop = deadline(optimValues, limit)
    persistent t0
    if isempty(t0) || optimValues.iteration == 0, t0 = tic; end
    stop = toc(t0) > limit;
end
