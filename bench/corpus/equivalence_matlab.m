% Evaluate every generated MATLAB problem at the probe points from probes.json
% and write matlab_probes.json for equivalence.py to compare.
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, 'matlab'));
probes = jsondecode(fileread(fullfile(here, 'probes.json')));
names = fieldnames(probes);
out = struct();
for k = 1:numel(names)
    nm = names{k};
    p = feval(nm);
    pts = probes.(nm);
    res = cell(1, numel(pts));
    for j = 1:numel(pts)
        x = pts(j).x(:);
        r = struct();
        r.f = p.f(x); r.grad = p.grad(x)'; r.c = p.c(x)'; r.jac = p.jac(x);
        res{j} = r;
    end
    out.(nm) = res;
end
fid = fopen(fullfile(here, 'matlab_probes.json'), 'w'); fprintf(fid, '%s', jsonencode(out)); fclose(fid);
disp('wrote matlab_probes.json');
