//! Restoration on the existing slack formulation; see docs/12.
use super::*;

const RHO: f64 = 1000.0;
const KAPPA_F: f64 = 0.999;
const KAPPA_RESTO: f64 = 0.9;

pub(super) struct Recovery {
    pub point: Point,
    pub grad: Vec<f64>,
    pub lambda: Vec<f64>,
    pub zl: Vec<f64>,
    pub zu: Vec<f64>,
    pub steps: usize,
    pub exit: Option<ExitFlag>,
}

/// Exact minimizer of rho*(p+n)-mu*log(p*n), with p-n=r.
/// Returns objective, envelope gradient and inverse envelope curvature.
fn elastic(r: f64, mu: f64) -> (f64, f64, f64) {
    let a = mu / RHO;
    let q = a.hypot(r);
    let small = 0.5 * (a + a * (a / (q + r.abs())));
    let large = r.abs() + small;
    (
        RHO * (large + small) - mu * (large.ln() + small.ln()),
        RHO * (r / (q + a)),
        (q / RHO) * ((q + a) / a),
    )
}

impl<P: Nlp + ?Sized> Solver<'_, P> {
    fn recovery_budget_hit(&self) -> bool {
        self.opts
            .max_evaluations
            .is_some_and(|limit| mincon_core::EvalCounters::get(&self.eval.counters().f) >= limit)
            || self
                .opts
                .max_seconds
                .is_some_and(|limit| self.start.elapsed().as_secs_f64() >= limit)
    }

    fn recovery_derivatives(&mut self, point: &Point) -> Result<Vec<f64>, EvalError> {
        let mut grad = vec![0.0; self.n];
        self.eval
            .grad(&point.v[..self.n], point.f / self.d_f, &mut grad)?;
        for g in &mut grad {
            *g *= self.d_f;
        }
        self.refresh_jacobian(&point.v[..self.n], &point.c)?;
        Ok(grad)
    }

    fn residual_l1(&self, state: &Recovery, mu: f64) -> f64 {
        let mut al = vec![0.0; self.nv];
        self.a_times(&state.lambda, &mut al);
        let mut sum = state.point.theta;
        for j in 0..self.nv {
            let gf = if j < self.n { state.grad[j] } else { 0.0 };
            sum += (gf + al[j] - state.zl[j] + state.zu[j]).abs();
            if self.has_l[j] {
                sum += ((state.point.v[j] - self.v_l[j]) * state.zl[j] - mu).abs();
            }
            if self.has_u[j] {
                sum += ((self.v_u[j] - state.point.v[j]) * state.zu[j] - mu).abs();
            }
        }
        sum
    }

    fn record_recovery(
        &self,
        s: &Recovery,
        mu: f64,
        iter: usize,
        step: f64,
        alpha: f64,
        trace: &mut Vec<IterationRecord>,
    ) {
        if self.opts.record_trace {
            trace.push(IterationRecord {
                iter,
                f_count: mincon_core::EvalCounters::get(&self.eval.counters().f),
                f: s.point.f / self.d_f,
                constraint_violation: self.user_violation(&s.point.v, &s.point.c),
                optimality: self
                    .optimality(&s.point, &s.grad, &s.lambda, &s.zl, &s.zu, mu)
                    .0,
                step_norm: step,
                alpha,
                mu,
                delta_w: 0.0,
                delta_c: 0.0,
                in_restoration: true,
                soc_count: 0,
            });
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn restore(
        &mut self,
        point: &Point,
        grad: &[f64],
        lambda: &[f64],
        zl: &[f64],
        zu: &[f64],
        mu: f64,
        filter: &mut Filter,
        remaining: usize,
        base_iter: usize,
        trace: &mut Vec<IterationRecord>,
        try_soft: bool,
    ) -> Result<Recovery, EvalError> {
        let mut state = Recovery {
            point: point.clone(),
            grad: grad.to_vec(),
            lambda: lambda.to_vec(),
            zl: zl.to_vec(),
            zu: zu.to_vec(),
            steps: 0,
            exit: None,
        };
        let theta_r = point.theta;
        filter.augment(point.theta, point.phi);
        self.notes.push("Entered feasibility restoration.".into());
        self.record_recovery(&state, mu, base_iter, 0.0, 0.0, trace);
        if try_soft {
            self.soft_restoration(&mut state, mu, filter, remaining, base_iter, trace)?;
            self.refresh_jacobian(&state.point.v[..self.n], &state.point.c)?;
            if state.exit.is_some()
                || (state.steps > 0 && !filter.is_blocked(state.point.theta, state.point.phi))
            {
                return Ok(state);
            }
            if let Hess::Bfgs(b) = &mut self.hess {
                b.reset(1.0);
                self.notes.push("Reset the BFGS model once after unsuccessful soft restoration; retrying with the same residual and filter safeguards.".into());
                self.soft_restoration(&mut state, mu, filter, remaining, base_iter, trace)?;
                self.refresh_jacobian(&state.point.v[..self.n], &state.point.c)?;
                if state.exit.is_some()
                    || (state.steps > 0 && !filter.is_blocked(state.point.theta, state.point.phi))
                {
                    return Ok(state);
                }
            }
        }
        // A rejected soft trial may have installed its Jacobian. Restore it.
        self.refresh_jacobian(&state.point.v[..self.n], &state.point.c)?;
        if self.m == 0 || theta_r <= self.opts.tol.optimality {
            self.notes.push("Restoration cannot reduce near-zero infeasibility; stationarity remains unverified.".into());
            state.exit = Some(ExitFlag::NumericalFailure);
            return Ok(state);
        }
        self.robust_restoration(&mut state, mu, theta_r, filter, remaining, base_iter, trace)?;
        Ok(state)
    }

    #[allow(clippy::too_many_arguments)]
    fn soft_restoration(
        &mut self,
        state: &mut Recovery,
        mu: f64,
        filter: &Filter,
        remaining: usize,
        base_iter: usize,
        trace: &mut Vec<IterationRecord>,
    ) -> Result<(), EvalError> {
        let tau = self.opts.tau_min.max(1.0 - mu);
        while state.steps < remaining && !self.recovery_budget_hit() {
            let before = self.residual_l1(state, mu);
            let mut sigma = vec![0.0; self.nv];
            for j in 0..self.nv {
                if self.has_l[j] {
                    sigma[j] += state.zl[j] / (state.point.v[j] - self.v_l[j]);
                }
                if self.has_u[j] {
                    sigma[j] += state.zu[j] / (self.v_u[j] - state.point.v[j]);
                }
            }
            let h = self
                .hessian_values(&state.point.v[..self.n], &state.lambda)?
                .to_vec();
            // Soft restoration never relies on an unchecked inertia-free step.
            if self
                .kkt
                .factor_with_correction(
                    &h,
                    &self.jac_t_values,
                    &sigma,
                    mu,
                    mincon_core::RegularizationMode::Inertia,
                    &self.correction,
                )
                .is_err()
            {
                self.notes
                    .push("Soft restoration could not obtain certified KKT inertia.".into());
                break;
            }
            let mut g = vec![0.0; self.nv];
            let mut al = vec![0.0; self.nv];
            self.grad_phi(&state.grad, &state.point.v, mu, &mut g);
            self.a_times(&state.lambda, &mut al);
            let rhs = self.kkt.rhs_mut();
            for j in 0..self.nv {
                rhs[j] = -g[j] - al[j];
            }
            for i in 0..self.m {
                rhs[self.nv + i] = -state.point.c_hat[i];
            }
            if self.kkt.solve_scratch(self.opts.refinement_steps).is_err() {
                break;
            }
            let dv = self.kkt.sol()[..self.nv].to_vec();
            let dl = self.kkt.sol()[self.nv..].to_vec();
            let mut dzl = vec![0.0; self.nv];
            let mut dzu = vec![0.0; self.nv];
            for j in 0..self.nv {
                if self.has_l[j] {
                    let d = state.point.v[j] - self.v_l[j];
                    dzl[j] = mu / d - state.zl[j] - state.zl[j] / d * dv[j];
                }
                if self.has_u[j] {
                    let d = self.v_u[j] - state.point.v[j];
                    dzu[j] = mu / d - state.zu[j] + state.zu[j] / d * dv[j];
                }
            }
            let beta = self
                .fraction_to_boundary(&state.point.v, &dv, tau)
                .min(fraction_to_boundary_dual(&state.zl, &dzl, tau))
                .min(fraction_to_boundary_dual(&state.zu, &dzu, tau));
            let v: Vec<_> = state
                .point
                .v
                .iter()
                .zip(&dv)
                .map(|(x, d)| x + beta * d)
                .collect();
            let candidate = match self.evaluate(&v, mu) {
                Ok(p) => p,
                Err(EvalError::UserAbort) => return Err(EvalError::UserAbort),
                Err(_) => break,
            };
            let grad = match self.recovery_derivatives(&candidate) {
                Ok(g) => g,
                Err(EvalError::UserAbort) => return Err(EvalError::UserAbort),
                Err(_) => break,
            };
            let trial = Recovery {
                point: candidate,
                grad,
                lambda: state
                    .lambda
                    .iter()
                    .zip(&dl)
                    .map(|(x, d)| x + beta * d)
                    .collect(),
                zl: state
                    .zl
                    .iter()
                    .zip(&dzl)
                    .map(|(x, d)| x + beta * d)
                    .collect(),
                zu: state
                    .zu
                    .iter()
                    .zip(&dzu)
                    .map(|(x, d)| x + beta * d)
                    .collect(),
                steps: state.steps + 1,
                exit: None,
            };
            let after = self.residual_l1(&trial, mu);
            if !after.is_finite() || after > KAPPA_F * before {
                self.notes.push(format!("Soft restoration rejected residual {before:.3e} -> {after:.3e}, common step {beta:.3e}."));
                break;
            }
            *state = trial;
            let step = dv.iter().fold(0.0_f64, |s, d| s.max((beta * d).abs()));
            self.record_recovery(state, mu, base_iter + state.steps, step, beta, trace);
            if !filter.is_blocked(state.point.theta, state.point.phi) {
                self.notes.push(format!("Soft restoration re-entered after {} steps; barrier residual fell from {before:.3e} to {after:.3e} on the last step.", state.steps));
                return Ok(());
            }
        }
        if state.steps >= remaining || self.recovery_budget_hit() {
            state.exit = Some(ExitFlag::MaxReached);
        }
        Ok(())
    }

    fn restoration_objective(&self, point: &Point, reference: &[f64], dr: &[f64], mu: f64) -> f64 {
        let proximity: f64 = point
            .v
            .iter()
            .zip(reference)
            .zip(dr)
            .map(|((x, r), d)| (d * (x - r)).powi(2))
            .sum();
        point.c_hat.iter().map(|r| elastic(*r, mu).0).sum::<f64>()
            + 0.5 * mu.sqrt() * proximity
            + self.barrier_term(&point.v, mu)
    }

    #[allow(clippy::too_many_arguments)]
    fn robust_restoration(
        &mut self,
        state: &mut Recovery,
        original_mu: f64,
        theta_r: f64,
        filter: &Filter,
        remaining: usize,
        base_iter: usize,
        trace: &mut Vec<IterationRecord>,
    ) -> Result<(), EvalError> {
        let reference = state.point.v.clone();
        let dr: Vec<_> = reference.iter().map(|x| 1.0 / x.abs().max(1.0)).collect();
        let mut mu = state
            .point
            .c_hat
            .iter()
            .fold(original_mu, |a, r| a.max(r.abs()));
        let floor = self.opts.tol.optimality / 10.0;
        let zero_h = vec![0.0; self.hess.pattern().nnz()];
        self.notes.push("Using the reduced elastic feasibility phase (analytical elastic minimization, Gauss-Newton curvature).".into());
        while state.steps < remaining && !self.recovery_budget_hit() {
            self.refresh_jacobian(&state.point.v[..self.n], &state.point.c)?;
            let mut lm = vec![0.0; self.m];
            let mut t = vec![0.0; self.m];
            for i in 0..self.m {
                let (_, l, ti) = elastic(state.point.c_hat[i], mu);
                lm[i] = l;
                t[i] = ti;
            }
            let mut g = vec![0.0; self.nv];
            self.a_times(&lm, &mut g);
            let mut sigma = vec![0.0; self.nv];
            let mut stationarity = vec![0.0; self.nv];
            for j in 0..self.nv {
                stationarity[j] = g[j] / RHO;
                let prox = mu.sqrt() * dr[j] * dr[j];
                g[j] += prox * (state.point.v[j] - reference[j]);
                sigma[j] = prox;
                if self.has_l[j] {
                    let d = state.point.v[j] - self.v_l[j];
                    g[j] -= mu / d;
                    sigma[j] += mu / (d * d);
                    stationarity[j] -= mu / (RHO * d);
                }
                if self.has_u[j] {
                    let d = self.v_u[j] - state.point.v[j];
                    g[j] += mu / d;
                    sigma[j] += mu / (d * d);
                    stationarity[j] += mu / (RHO * d);
                }
            }
            let norm = g.iter().fold(0.0_f64, |a, x| a.max(x.abs()));
            let stat = stationarity.iter().fold(0.0_f64, |a, x| a.max(x.abs()));
            if mu <= floor
                && stat <= self.opts.tol.optimality
                && self.user_violation(&state.point.v, &state.point.c) > self.opts.tol.feasibility
            {
                self.notes.push(format!("Restoration reached stationary positive violation (normalized residual {stat:.3e}); this is a local diagnostic, not a proof of infeasibility or a second-order minimum."));
                state.exit = Some(ExitFlag::LocallyInfeasible);
                return Ok(());
            }
            if norm <= self.barrier.kappa_eps * mu && mu > floor {
                mu = floor.max((self.barrier.kappa_mu * mu).min(mu.powf(self.barrier.theta_mu)));
                // A barrier update is work too; prevents uncharged unbounded loops.
                state.steps += 1;
                self.record_recovery(state, original_mu, base_iter + state.steps, 0.0, 0.0, trace);
                continue;
            }
            if self
                .kkt
                .factor_restoration(
                    &zero_h,
                    &self.jac_t_values,
                    &sigma,
                    &t,
                    mu,
                    &self.correction,
                )
                .is_err()
            {
                break;
            }
            let rhs = self.kkt.rhs_mut();
            rhs.fill(0.0);
            for j in 0..self.nv {
                rhs[j] = -g[j];
            }
            if self.kkt.solve_scratch(self.opts.refinement_steps).is_err() {
                break;
            }
            let dv = self.kkt.sol()[..self.nv].to_vec();
            let slope: f64 = g.iter().zip(&dv).map(|(g, d)| g * d).sum();
            if !slope.is_finite() || slope >= 0.0 {
                break;
            }
            let mut alpha =
                self.fraction_to_boundary(&state.point.v, &dv, self.opts.tau_min.max(1.0 - mu));
            let merit = self.restoration_objective(&state.point, &reference, &dr, mu);
            let mut accepted = None;
            for _ in 0..filter.params().max_backtracks {
                if self.recovery_budget_hit() {
                    break;
                }
                let v: Vec<_> = state
                    .point
                    .v
                    .iter()
                    .zip(&dv)
                    .map(|(x, d)| x + alpha * d)
                    .collect();
                match self.evaluate(&v, original_mu) {
                    Ok(p) => {
                        let trial_merit = self.restoration_objective(&p, &reference, &dr, mu);
                        if trial_merit.is_finite()
                            && trial_merit <= merit + filter.params().eta_phi * alpha * slope
                        {
                            match self.recovery_derivatives(&p) {
                                Ok(grad) => {
                                    accepted = Some((p, grad));
                                    break;
                                }
                                Err(EvalError::UserAbort) => return Err(EvalError::UserAbort),
                                Err(_) => {}
                            }
                        }
                    }
                    Err(EvalError::UserAbort) => return Err(EvalError::UserAbort),
                    Err(_) => {}
                }
                alpha *= filter.params().backtrack;
            }
            let Some((p, grad)) = accepted else {
                self.refresh_jacobian(&state.point.v[..self.n], &state.point.c)?;
                break;
            };
            state.point = p;
            state.grad = grad;
            state.steps += 1;
            let step = dv.iter().fold(0.0_f64, |a, d| a.max((alpha * d).abs()));
            self.record_recovery(
                state,
                original_mu,
                base_iter + state.steps,
                step,
                alpha,
                trace,
            );
            if state.point.theta <= KAPPA_RESTO * theta_r
                && !filter.is_blocked(state.point.theta, state.point.phi)
            {
                state.lambda.fill(0.0);
                for j in 0..self.nv {
                    state.zl[j] = if self.has_l[j] {
                        original_mu / (state.point.v[j] - self.v_l[j])
                    } else {
                        0.0
                    };
                    state.zu[j] = if self.has_u[j] {
                        original_mu / (self.v_u[j] - state.point.v[j])
                    } else {
                        0.0
                    };
                }
                self.notes.push(format!(
                    "Elastic restoration re-entered: violation {theta_r:.3e} -> {:.3e}.",
                    state.point.theta
                ));
                return Ok(());
            }
        }
        state.exit = Some(if state.steps >= remaining || self.recovery_budget_hit() {
            ExitFlag::MaxReached
        } else {
            ExitFlag::NumericalFailure
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DerivativeDomain {
        abort: bool,
    }
    impl Nlp for DerivativeDomain {
        fn dims(&self) -> mincon_core::NlpDims {
            mincon_core::NlpDims { n: 1, m: 1 }
        }
        fn x_bounds(&self) -> (&[f64], &[f64]) {
            (&[0.0], &[2.0])
        }
        fn c_bounds(&self) -> (&[f64], &[f64]) {
            (&[0.0], &[0.0])
        }
        fn x0(&self) -> &[f64] {
            &[0.1]
        }
        fn capabilities(&self) -> mincon_core::Capabilities {
            mincon_core::Capabilities {
                gradient: true,
                ..Default::default()
            }
        }
        fn objective(&self, x: &[f64]) -> Result<f64, EvalError> {
            Ok(x[0] * x[0])
        }
        fn constraints(&self, x: &[f64], c: &mut [f64]) -> Result<(), EvalError> {
            c[0] = x[0] * x[0] - 0.25;
            Ok(())
        }
        fn gradient(&self, x: &[f64], g: &mut [f64]) -> Result<(), EvalError> {
            if x[0] > 0.4 {
                return Err(if self.abort {
                    EvalError::UserAbort
                } else {
                    EvalError::OutOfDomain("derivative domain".into())
                });
            }
            g[0] = 2.0 * x[0];
            Ok(())
        }
    }

    #[test]
    fn restoration_retries_derivative_failures_but_propagates_user_abort() {
        for abort in [false, true] {
            let p = DerivativeDomain { abort };
            let mut solver = Solver::new(&p, &Options::default()).unwrap();
            let point = solver.evaluate(&[0.1], 0.1).unwrap();
            let grad = solver.recovery_derivatives(&point).unwrap();
            let mut filter = Filter::new(point.theta, FilterParams::default());
            let recovered = solver.restore(
                &point,
                &grad,
                &[0.0],
                &[1.0],
                &[1.0],
                0.1,
                &mut filter,
                100,
                0,
                &mut vec![],
                false,
            );
            if abort {
                assert!(matches!(recovered, Err(EvalError::UserAbort)));
                let calls = mincon_core::EvalCounters::get(&solver.eval.counters().g);
                let report = solver.finish(
                    ExitFlag::StoppedByUser,
                    0,
                    point,
                    &grad,
                    vec![0.0],
                    vec![1.0],
                    vec![1.0],
                    vec![],
                );
                assert_eq!(report.g_evals, calls);
                assert_eq!(report.solution.x, vec![0.1]);
                assert!(!report.exit_flag.is_success());
            } else {
                let r = recovered.unwrap();
                assert!(r.exit.is_none());
                assert!((r.point.v[0] * r.point.v[0] - 0.25).abs() <= 0.9 * 0.24);
                assert!(r.point.v[0] <= 0.4);
                assert!(mincon_core::EvalCounters::get(&solver.eval.counters().failed) > 0);
            }
        }
    }

    fn feasibility_fixture() -> mincon_testset::TestProblem {
        mincon_testset::TestProblem {
            name: "curved feasibility",
            n: 1,
            m: 1,
            x0: vec![0.1],
            xl: vec![0.0],
            xu: vec![2.0],
            cl: vec![0.0],
            cu: vec![0.0],
            f: |x| if x[0] > 0.8 { f64::NAN } else { x[0] * x[0] },
            c: |x, c| c[0] = x[0] * x[0] - 0.25,
            f_opt: Some(0.25),
            expect: mincon_testset::Expect::Optimum,
            notes: "positive solution x=0.5",
        }
    }

    #[test]
    fn full_restoration_reenters_and_retreats_from_invalid_trials() {
        let p = feasibility_fixture();
        let nlp = p.as_nlp();
        let mut solver = Solver::new(&nlp, &Options::default()).unwrap();
        let point = solver.evaluate(&[0.1], 0.1).unwrap();
        let grad = solver.recovery_derivatives(&point).unwrap();
        let mut filter = Filter::new(point.theta, FilterParams::default());
        let mut trace = vec![];
        let r = solver
            .restore(
                &point,
                &grad,
                &[0.0],
                &[1.0],
                &[1.0],
                0.1,
                &mut filter,
                100,
                0,
                &mut trace,
                false,
            )
            .unwrap();
        assert!(r.exit.is_none());
        assert!(!filter.is_blocked(r.point.theta, r.point.phi));
        assert!(p.violation(&r.point.v) <= 0.9 * p.violation(&point.v));
        assert!(mincon_core::EvalCounters::get(&solver.eval.counters().failed) > 0);
        assert!(r.point.v[0] >= 0.0 && r.point.v[0] <= 0.8);
        assert!(trace.iter().all(|t| t.in_restoration));
    }

    #[test]
    fn restoration_respects_the_parent_iteration_and_evaluation_limits() {
        let p = feasibility_fixture();
        let nlp = p.as_nlp();
        let mut solver = Solver::new(&nlp, &Options::default()).unwrap();
        let point = solver.evaluate(&[0.1], 0.1).unwrap();
        let grad = solver.recovery_derivatives(&point).unwrap();
        let mut filter = Filter::new(point.theta, FilterParams::default());
        let r = solver
            .restore(
                &point,
                &grad,
                &[0.0],
                &[1.0],
                &[1.0],
                0.1,
                &mut filter,
                0,
                0,
                &mut vec![],
                false,
            )
            .unwrap();
        assert_eq!(r.exit, Some(ExitFlag::MaxReached));
        assert_eq!(r.steps, 0);
        solver.opts.max_evaluations =
            Some(mincon_core::EvalCounters::get(&solver.eval.counters().f));
        let r = solver
            .restore(
                &point,
                &grad,
                &[0.0],
                &[1.0],
                &[1.0],
                0.1,
                &mut filter,
                100,
                0,
                &mut vec![],
                true,
            )
            .unwrap();
        assert_eq!(r.exit, Some(ExitFlag::MaxReached));
        assert_eq!(r.steps, 0);
    }
    #[test]
    fn poisoned_bfgs_model_recovers_on_an_analytic_quadratic() {
        let p = mincon_testset::TestProblem {
            name: "quadratic",
            n: 1,
            m: 0,
            x0: vec![1.0],
            xl: vec![f64::NEG_INFINITY],
            xu: vec![f64::INFINITY],
            cl: vec![],
            cu: vec![],
            f: |x| 0.5 * x[0] * x[0],
            c: |_, _| {},
            f_opt: Some(0.0),
            expect: mincon_testset::Expect::Optimum,
            notes: "analytic",
        };
        let nlp = p.as_nlp();
        let mut solver = Solver::new(&nlp, &Options::default()).unwrap();
        solver.eval.escalate_accuracy();
        if let Hess::Bfgs(b) = &mut solver.hess {
            b.reset(1e24);
        }
        let point = solver.evaluate(&[1.0], 0.1).unwrap();
        let mut filter = Filter::new(0.0, FilterParams::default());
        let r = solver
            .restore(
                &point,
                &[1.0],
                &[],
                &[0.0],
                &[0.0],
                0.1,
                &mut filter,
                20,
                0,
                &mut vec![],
                true,
            )
            .unwrap();
        assert!(r.exit.is_none());
        assert!(r.point.v[0].abs() < 1e-8);
    }
    #[test]
    fn eliminated_elastics_match_analytic_kkt_and_envelope_derivatives() {
        for r in [-10.0_f64, -0.001, 0.0, 0.001, 10.0] {
            let mu = 0.2;
            let (_, l, t) = elastic(r, mu);
            let p = mu / (RHO - l);
            let n = mu / (RHO + l);
            assert!((p - n - r).abs() < 1e-8);
            assert!(p > 0.0 && n > 0.0 && t > 0.0);
            // Independent differentiation of c(lambda)=2*mu*lambda/(rho^2-lambda^2).
            let inverse = 2.0 * mu * (RHO * RHO + l * l) / (RHO * RHO - l * l).powi(2);
            assert!((t - inverse).abs() / t < 1e-8);
        }
        assert!((elastic(0.0, 0.2).2 - 2.0 * 0.2 / (RHO * RHO)).abs() < 1e-20);
    }
}
