//! The Fletcher–Leyffer filter and the line-search acceptance rules.
//!
//! # Why a filter rather than a penalty function
//!
//! A merit function `phi(x) + nu * ||c(x)||` needs a penalty parameter `nu`,
//! and choosing it is a genuinely hard sub-problem: too small and infeasible
//! points look attractive, too large and the method crawls. `fmincon`'s SQP and
//! interior-point algorithms both use merit functions and both spend real
//! effort on the `nu` update.
//!
//! A filter sidesteps the parameter entirely by treating the problem as
//! bi-objective: a trial point is acceptable if it improves either the barrier
//! objective `phi` or the constraint violation `theta` relative to *every*
//! point already in the filter. There is no parameter to tune, which is exactly
//! the kind of automatic behaviour we are competing on.
//!
//! The cost is that a filter needs care to avoid accepting a sequence that
//! converges to an infeasible point, which is what the switching condition and
//! the feasibility restoration phase are for.
//!
//! Constants follow Wächter and Biegler (2006), Section 2.3.

/// One entry: a `(theta, phi)` pair that later iterates must beat.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Entry {
    theta: f64,
    phi: f64,
}

/// Tunables of the filter line search.
#[derive(Debug, Clone, Copy)]
pub struct FilterParams {
    /// `gamma_theta`: margin by which a new entry dominates in `theta`.
    pub gamma_theta: f64,
    /// `gamma_phi`: margin by which a new entry dominates in `phi`.
    pub gamma_phi: f64,
    /// `delta`: coefficient in the switching condition.
    pub delta: f64,
    /// `s_theta`: exponent on `theta` in the switching condition.
    pub s_theta: f64,
    /// `s_phi`: exponent on the directional derivative in the switching condition.
    pub s_phi: f64,
    /// `eta_phi`: Armijo relaxation.
    pub eta_phi: f64,
    /// `gamma_alpha`: safety factor on the minimum step length.
    pub gamma_alpha: f64,
    /// `theta_max` multiplier applied to the initial violation.
    pub theta_max_factor: f64,
    /// `theta_min` multiplier applied to the initial violation.
    pub theta_min_factor: f64,
    /// Backtracking factor.
    pub backtrack: f64,
    /// Maximum backtracking steps before declaring the line search failed.
    pub max_backtracks: usize,
}

impl Default for FilterParams {
    fn default() -> Self {
        Self {
            gamma_theta: 1e-5,
            gamma_phi: 1e-5,
            delta: 1.0,
            s_theta: 1.1,
            s_phi: 2.3,
            eta_phi: 1e-8,
            gamma_alpha: 0.05,
            theta_max_factor: 1e4,
            theta_min_factor: 1e-4,
            backtrack: 0.5,
            max_backtracks: 40,
        }
    }
}

/// Why a trial point was accepted, which determines whether the filter grows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Acceptance {
    /// The switching condition held and Armijo was satisfied: a genuine
    /// objective-reduction step. The filter is **not** augmented, which is what
    /// preserves fast local convergence.
    ArmijoDescent,
    /// Sufficient reduction in `theta` or in `phi`. The filter **is** augmented.
    SufficientDecrease,
    /// Rejected.
    Rejected,
}

/// The filter.
#[derive(Debug, Clone)]
pub struct Filter {
    entries: Vec<Entry>,
    params: FilterParams,
    theta_max: f64,
    theta_min: f64,
}

impl Filter {
    /// Initialize from the constraint violation at the starting point.
    #[must_use]
    pub fn new(theta0: f64, params: FilterParams) -> Self {
        let base = theta0.max(1.0);
        Self {
            entries: Vec::new(),
            params,
            theta_max: params.theta_max_factor * base,
            theta_min: params.theta_min_factor * base,
        }
    }

    /// Tunables in use.
    #[must_use]
    pub fn params(&self) -> &FilterParams {
        &self.params
    }
    /// `theta_max`: violation above which a point is unconditionally rejected.
    #[must_use]
    pub fn theta_max(&self) -> f64 {
        self.theta_max
    }
    /// `theta_min`: below this the switching condition may fire.
    #[must_use]
    pub fn theta_min(&self) -> f64 {
        self.theta_min
    }
    /// Number of stored entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    /// Whether the filter holds no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Clear the filter and shrink `theta_max`. The acceleration heuristic from
    /// Wächter–Biegler Section 3.4 Case I, used when repeated trial steps are
    /// being rejected by the filter rather than by the objective.
    pub fn reset_and_tighten(&mut self, current_theta: f64) {
        self.entries.clear();
        self.theta_max = (0.1 * self.theta_max).max(current_theta * 1.5);
    }

    /// Whether `(theta, phi)` is blocked by the filter or by `theta_max`.
    #[must_use]
    pub fn is_blocked(&self, theta: f64, phi: f64) -> bool {
        if !theta.is_finite() || !phi.is_finite() {
            return true;
        }
        if theta >= self.theta_max {
            return true;
        }
        self.entries
            .iter()
            .any(|e| theta >= e.theta && phi >= e.phi)
    }

    /// Add `(theta, phi)` with the standard margins, dropping entries the new
    /// one dominates.
    pub fn augment(&mut self, theta: f64, phi: f64) {
        let e = Entry {
            theta: (1.0 - self.params.gamma_theta) * theta,
            phi: phi - self.params.gamma_phi * theta,
        };
        self.entries
            .retain(|old| !(old.theta >= e.theta && old.phi >= e.phi));
        self.entries.push(e);
    }

    /// The switching condition: is this step big enough, and downhill enough,
    /// to be judged on objective decrease alone?
    #[must_use]
    pub fn switching_condition(&self, alpha: f64, dphi: f64, theta: f64) -> bool {
        if dphi >= 0.0 {
            return false;
        }
        let p = &self.params;
        alpha * (-dphi).powf(p.s_phi) > p.delta * theta.powf(p.s_theta) && theta <= self.theta_min
    }

    /// Minimum step length before the line search gives up and restoration
    /// takes over. Wächter–Biegler equation (23).
    #[must_use]
    pub fn min_step_size(&self, dphi: f64, theta: f64) -> f64 {
        let p = &self.params;
        if dphi >= 0.0 {
            return p.gamma_alpha * p.gamma_theta;
        }
        let a = p.gamma_theta;
        let b = p.gamma_phi * theta / (-dphi);
        let c = p.delta * theta.powf(p.s_theta) / (-dphi).powf(p.s_phi);
        let m = if theta <= self.theta_min {
            a.min(b).min(c)
        } else {
            a.min(b)
        };
        (p.gamma_alpha * m).max(1e-16)
    }

    /// Test a trial point.
    ///
    /// * `theta0`, `phi0` — at the current iterate.
    /// * `theta`, `phi` — at the trial point.
    /// * `alpha` — the step length tried.
    /// * `dphi` — directional derivative of the barrier objective.
    #[must_use]
    pub fn evaluate(
        &self,
        theta0: f64,
        phi0: f64,
        theta: f64,
        phi: f64,
        alpha: f64,
        dphi: f64,
    ) -> Acceptance {
        if self.is_blocked(theta, phi) {
            return Acceptance::Rejected;
        }
        let p = &self.params;
        if self.switching_condition(alpha, dphi, theta0) {
            // Objective-reduction regime: Armijo alone decides.
            if phi <= phi0 + p.eta_phi * alpha * dphi {
                return Acceptance::ArmijoDescent;
            }
            return Acceptance::Rejected;
        }
        if theta <= (1.0 - p.gamma_theta) * theta0 || phi <= phi0 - p.gamma_phi * theta0 {
            return Acceptance::SufficientDecrease;
        }
        Acceptance::Rejected
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_empty_filter_blocks_only_very_infeasible_points() {
        let f = Filter::new(1.0, FilterParams::default());
        assert!(!f.is_blocked(0.5, 100.0));
        assert!(f.is_blocked(f.theta_max() + 1.0, -1e6));
    }

    #[test]
    fn dominated_points_are_blocked_and_dominating_ones_are_not() {
        let mut f = Filter::new(1.0, FilterParams::default());
        f.augment(1.0, 10.0);
        assert!(f.is_blocked(2.0, 20.0), "worse in both must be blocked");
        assert!(!f.is_blocked(0.1, 20.0), "better theta must pass");
        assert!(!f.is_blocked(2.0, 1.0), "better phi must pass");
    }

    #[test]
    fn augmenting_prunes_dominated_entries() {
        let mut f = Filter::new(1.0, FilterParams::default());
        f.augment(2.0, 20.0);
        f.augment(3.0, 30.0);
        assert_eq!(f.len(), 2);
        // A point better in both dominates and replaces both.
        f.augment(0.5, 5.0);
        assert_eq!(f.len(), 1, "dominated entries should be pruned");
    }

    #[test]
    fn non_finite_trial_points_are_always_blocked() {
        let f = Filter::new(1.0, FilterParams::default());
        assert!(f.is_blocked(f64::NAN, 1.0));
        assert!(f.is_blocked(1.0, f64::INFINITY));
    }

    #[test]
    fn switching_condition_needs_descent_and_near_feasibility() {
        let f = Filter::new(1.0, FilterParams::default());
        // Uphill: never switches.
        assert!(!f.switching_condition(1.0, 1.0, 1e-8));
        // Downhill and nearly feasible: switches.
        assert!(f.switching_condition(1.0, -1.0, 1e-10));
        // Downhill but far from feasible: does not switch, so feasibility
        // progress is what gets judged.
        assert!(!f.switching_condition(1.0, -1.0, 1e6));
    }

    #[test]
    fn armijo_regime_accepts_a_real_objective_decrease() {
        let f = Filter::new(1.0, FilterParams::default());
        let a = f.evaluate(1e-12, 10.0, 1e-12, 9.0, 1.0, -1.0);
        assert_eq!(a, Acceptance::ArmijoDescent);
    }

    #[test]
    fn armijo_regime_rejects_an_insufficient_decrease() {
        let f = Filter::new(1.0, FilterParams::default());
        // phi barely moves against a steep predicted decrease.
        let a = f.evaluate(1e-12, 10.0, 1e-12, 10.0 + 1e-9, 1.0, -1.0);
        assert_eq!(a, Acceptance::Rejected);
    }

    #[test]
    fn sufficient_decrease_in_theta_is_accepted_when_not_switching() {
        let f = Filter::new(1.0, FilterParams::default());
        // theta0 large so the switching condition fails; theta drops a lot.
        let a = f.evaluate(1.0, 10.0, 0.5, 10.5, 1.0, -1.0);
        assert_eq!(a, Acceptance::SufficientDecrease);
    }

    #[test]
    fn sufficient_decrease_in_phi_is_accepted_when_not_switching() {
        let f = Filter::new(1.0, FilterParams::default());
        let a = f.evaluate(1.0, 10.0, 1.0, 9.0, 1.0, -1.0);
        assert_eq!(a, Acceptance::SufficientDecrease);
    }

    #[test]
    fn no_progress_is_rejected() {
        let f = Filter::new(1.0, FilterParams::default());
        let a = f.evaluate(1.0, 10.0, 1.0, 10.0, 1.0, -1.0);
        assert_eq!(a, Acceptance::Rejected);
    }

    #[test]
    fn minimum_step_size_is_positive_and_shrinks_with_steeper_descent() {
        let f = Filter::new(1.0, FilterParams::default());
        let a = f.min_step_size(-1.0, 1e-6);
        let b = f.min_step_size(-1e6, 1e-6);
        assert!(a > 0.0 && b > 0.0);
        assert!(b < a, "a steeper direction should tolerate smaller steps");
    }

    #[test]
    fn reset_shrinks_theta_max() {
        let mut f = Filter::new(1.0, FilterParams::default());
        f.augment(1.0, 1.0);
        let before = f.theta_max();
        f.reset_and_tighten(1e-3);
        assert!(f.is_empty());
        assert!(f.theta_max() < before);
    }
}
