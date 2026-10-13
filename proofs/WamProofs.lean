import Mathlib

open scoped BigOperators
open Finset

namespace WamProofs

section RiskGap
variable {Ω : Type*} [Fintype Ω]

theorem wsum_sq (q y : Ω → ℝ) (hq : ∑ ω, q ω = 1) (c : ℝ) :
    ∑ ω, q ω * (y ω - c) ^ 2
      = ∑ ω, q ω * y ω ^ 2 - 2 * c * ∑ ω, q ω * y ω + c ^ 2 := by
  have h : ∀ ω, q ω * (y ω - c) ^ 2 = q ω * y ω ^ 2 - 2 * c * (q ω * y ω) + c ^ 2 * q ω := by
    intro ω; ring
  simp_rw [h]
  rw [sum_add_distrib, sum_sub_distrib, ← mul_sum, ← mul_sum, hq]
  ring

theorem cond_risk_gap (q y : Ω → ℝ) (hq : ∑ ω, q ω = 1) (c : ℝ) :
    ∑ ω, q ω * (y ω - c) ^ 2 - ∑ ω, q ω * (y ω - ∑ ω', q ω' * y ω') ^ 2
      = (∑ ω, q ω * y ω - c) ^ 2 := by
  rw [wsum_sq q y hq c, wsum_sq q y hq]
  ring

variable {H A : Type*} [Fintype H] [Fintype A]

theorem risk_gap (p : H → A → ℝ) (q : H → A → Ω → ℝ) (y : Ω → ℝ)
    (hq : ∀ h a, ∑ ω, q h a ω = 1) (g : H → ℝ) :
    (∑ h, ∑ a, p h a * ∑ ω, q h a ω * (y ω - g h) ^ 2)
      - (∑ h, ∑ a, p h a * ∑ ω, q h a ω * (y ω - ∑ ω', q h a ω' * y ω') ^ 2)
    = ∑ h, ∑ a, p h a * ((∑ ω, q h a ω * y ω) - g h) ^ 2 := by
  rw [← sum_sub_distrib]
  refine sum_congr rfl (fun h _ => ?_)
  rw [← sum_sub_distrib]
  refine sum_congr rfl (fun a _ => ?_)
  rw [← mul_sub, cond_risk_gap (q h a) y (hq h a) (g h)]

theorem blind_never_beats (p : H → A → ℝ) (hp : ∀ h a, 0 ≤ p h a) (q : H → A → Ω → ℝ)
    (y : Ω → ℝ) (hq : ∀ h a, ∑ ω, q h a ω = 1) (g : H → ℝ) :
    (∑ h, ∑ a, p h a * ∑ ω, q h a ω * (y ω - ∑ ω', q h a ω' * y ω') ^ 2)
      ≤ ∑ h, ∑ a, p h a * ∑ ω, q h a ω * (y ω - g h) ^ 2 := by
  have hgap := risk_gap p q y hq g
  have hnn : 0 ≤ ∑ h, ∑ a, p h a * ((∑ ω, q h a ω * y ω) - g h) ^ 2 :=
    sum_nonneg fun h _ => sum_nonneg fun a _ => mul_nonneg (hp h a) (sq_nonneg _)
  linarith

end RiskGap

section OffSupport

theorem offSupport_value {H A S : Type*} [AddCommGroup A] [Module ℝ A]
    [AddCommGroup S] [Module ℝ S]
    (f : H → A → S) (B : H → A →ₗ[ℝ] S) (π : H → A) (h : H) :
    f h (π h) + B h (π h - π h) = f h (π h) := by
  simp

theorem offSupport_risk {Ω H A S : Type*} [MeasurableSpace Ω] (μ : MeasureTheory.Measure Ω)
    [AddCommGroup A] [Module ℝ A] [AddCommGroup S] [Module ℝ S]
    (f : H → A → S) (B : H → A →ₗ[ℝ] S) (π : H → A)
    (Hr : Ω → H) (Ar : Ω → A) (Y : Ω → S) (loss : S → S → ℝ)
    (hπ : ∀ᵐ ω ∂μ, Ar ω = π (Hr ω)) :
    ∫ ω, loss (f (Hr ω) (Ar ω) + B (Hr ω) (Ar ω - π (Hr ω))) (Y ω) ∂μ
      = ∫ ω, loss (f (Hr ω) (Ar ω)) (Y ω) ∂μ := by
  refine MeasureTheory.integral_congr_ae ?_
  filter_upwards [hπ] with ω hω
  simp [hω]

theorem offSupport_deriv {A S : Type*} [NormedAddCommGroup A] [NormedSpace ℝ A]
    [NormedAddCommGroup S] [NormedSpace ℝ S]
    (g : A → S) (g' B : A →L[ℝ] S) (p a : A) (hg : HasFDerivAt g g' a) :
    HasFDerivAt (fun x => g x + B (x - p)) (g' + B) a := by
  have hB : HasFDerivAt (fun x => B x - B p) B a := B.hasFDerivAt.sub_const (B p)
  have hB' : HasFDerivAt (fun x => B (x - p)) B a := by simpa [map_sub] using hB
  exact hg.add hB'

theorem scaled_norm_unbounded {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (B : E) (hB : B ≠ 0) (M : ℝ) : ∃ c : ℝ, M < ‖c • B‖ := by
  have hpos : 0 < ‖B‖ := norm_pos_iff.mpr hB
  refine ⟨(|M| + 1) / ‖B‖, ?_⟩
  have hc : 0 < (|M| + 1) / ‖B‖ := by positivity
  rw [norm_smul, Real.norm_eq_abs, abs_of_pos hc]
  have hmul : (|M| + 1) / ‖B‖ * ‖B‖ = |M| + 1 := by field_simp
  rw [hmul]
  linarith [le_abs_self M]

theorem offSupport_unbounded {A S : Type*} [NormedAddCommGroup A] [NormedSpace ℝ A]
    [NormedAddCommGroup S] [NormedSpace ℝ S]
    (g' B : A →L[ℝ] S) (hB : B ≠ 0) (M : ℝ) : ∃ c : ℝ, M < ‖g' + c • B‖ := by
  obtain ⟨c, hc⟩ := scaled_norm_unbounded B hB (M + ‖g'‖)
  refine ⟨c, ?_⟩
  have htri := norm_sub_norm_le (c • B) (-g')
  rw [norm_neg, sub_neg_eq_add, add_comm] at htri
  linarith

end OffSupport

section AGM

theorem agm_unbounded_below (lam σ b : ℝ) (hl : 0 < lam) (hs : 0 < σ) (hb : 0 < b) (M : ℝ) :
    ∃ c : ℝ, -(lam * c ^ 2 * σ ^ 2 * b ^ 2) < M := by
  have hk : 0 < lam * σ ^ 2 * b ^ 2 := by positivity
  have hl0 : lam ≠ 0 := hl.ne'
  have hs0 : σ ≠ 0 := hs.ne'
  have hb0 : b ≠ 0 := hb.ne'
  refine ⟨Real.sqrt ((|M| + 1) / (lam * σ ^ 2 * b ^ 2)), ?_⟩
  have hsq : Real.sqrt ((|M| + 1) / (lam * σ ^ 2 * b ^ 2)) ^ 2
      = (|M| + 1) / (lam * σ ^ 2 * b ^ 2) := Real.sq_sqrt (by positivity)
  have hval : lam * Real.sqrt ((|M| + 1) / (lam * σ ^ 2 * b ^ 2)) ^ 2 * σ ^ 2 * b ^ 2
      = |M| + 1 := by
    rw [hsq]; field_simp; ring
  rw [hval]
  linarith [neg_abs_le M]

set_option synthInstance.maxHeartbeats 400000 in

theorem agm_zero_grad_at_blindness {Θ E : Type*} [NormedAddCommGroup Θ] [NormedSpace ℝ Θ]
    [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    (d : Θ → E) (d' : Θ →L[ℝ] E) (θ₀ : Θ) (hd : HasFDerivAt d d' θ₀) (h0 : d θ₀ = 0) :
    HasFDerivAt (fun θ => ‖d θ‖ ^ 2) (0 : Θ →L[ℝ] ℝ) θ₀ := by
  have hsq : HasFDerivAt (fun x : E => ‖x‖ ^ 2) (2 • innerSL ℝ (d θ₀)) (d θ₀) :=
    (hasStrictFDerivAt_norm_sq (d θ₀)).hasFDerivAt
  have h := hsq.comp θ₀ hd
  rw [h0] at h
  simpa using h

theorem agm_zero_grad_batch {ι Θ E : Type*} [NormedAddCommGroup Θ] [NormedSpace ℝ Θ]
    [NormedAddCommGroup E] [InnerProductSpace ℝ E] (s : Finset ι)
    (d : ι → Θ → E) (d' : ι → Θ →L[ℝ] E) (θ₀ : Θ)
    (hd : ∀ i ∈ s, HasFDerivAt (d i) (d' i) θ₀) (h0 : ∀ i ∈ s, d i θ₀ = 0) :
    HasFDerivAt (fun θ => ∑ i ∈ s, ‖d i θ‖ ^ 2) (0 : Θ →L[ℝ] ℝ) θ₀ := by
  have h := HasFDerivAt.sum (u := s) (A' := fun _ => (0 : Θ →L[ℝ] ℝ))
    (fun i hi => agm_zero_grad_at_blindness (d i) (d' i) θ₀ (hd i hi) (h0 i hi))
  simpa using h

theorem fd_second_moment {m n : ℕ} (J : Fin m → Fin n → ℝ) (σ : ℝ) :
    ∑ k, ∑ i, ∑ j, J k i * J k j * (if i = j then σ ^ 2 else 0)
      = σ ^ 2 * ∑ k, ∑ j, J k j ^ 2 := by
  rw [mul_sum]
  refine sum_congr rfl (fun k _ => ?_)
  rw [mul_sum]
  refine sum_congr rfl (fun i _ => ?_)
  simp [mul_ite]
  ring

theorem fd_expectation {Ω : Type*} {m n : ℕ} (E : (Ω → ℝ) →ₗ[ℝ] ℝ) (ε : Fin n → Ω → ℝ)
    (J : Fin m → Fin n → ℝ) (σ : ℝ)
    (hmom : ∀ i j, E (fun ω => ε i ω * ε j ω) = if i = j then σ ^ 2 else 0) :
    E (fun ω => ∑ k, (∑ j, J k j * ε j ω) ^ 2) = σ ^ 2 * ∑ k, ∑ j, J k j ^ 2 := by
  have hsq : (fun ω => ∑ k, (∑ j, J k j * ε j ω) ^ 2)
      = ∑ k, ∑ i, ∑ j, (J k i * J k j) • (fun ω => ε i ω * ε j ω) := by
    funext ω
    simp only [Finset.sum_apply, Pi.smul_apply, smul_eq_mul, sq, sum_mul_sum]
    refine sum_congr rfl (fun k _ => sum_congr rfl (fun i _ => sum_congr rfl (fun j _ => ?_)))
    ring
  rw [hsq]
  simp only [map_sum, map_smul, smul_eq_mul, hmom]
  exact fd_second_moment J σ

end AGM

section Stats

theorem did_cancels (a b c d nF nS : ℝ) :
    ((a + nF) - (b + nF)) - ((c + nS) - (d + nS)) = (a - b) - (c - d) := by
  ring

theorem ratio_not_invariant : ∃ a c n : ℝ, (c + n) / (a + n) ≠ c / a :=
  ⟨1, 0, 1, by norm_num⟩

theorem sum_range_real (n : ℕ) : ∑ k ∈ range n, (k : ℝ) = n * (n - 1) / 2 := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [sum_range_succ, ih]
    push_cast
    ring

theorem mean_rank_chance (n : ℕ) (hn : 0 < n) :
    (∑ k ∈ range n, (k : ℝ)) / n = (n - 1) / 2 := by
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast hn.ne'
  rw [sum_range_real]
  field_simp
  ring

end Stats

section Triangular

noncomputable def tri (m o : ℝ) : ℝ := if m ≤ o then m / o else (1 - m) / (1 - o)

theorem tri_self (o : ℝ) (h0 : 0 < o) : tri o o = 1 := by
  simp [tri, div_self h0.ne']

theorem tri_const_lower {lo hi o r : ℝ} (hlo : 0 < lo) (hhi : hi < 1)
    (ho : lo ≤ o ∧ o ≤ hi) (hr : lo ≤ r ∧ r ≤ hi) :
    min (lo / hi) ((1 - hi) / (1 - lo)) ≤ tri r o := by
  unfold tri
  split_ifs with h
  · apply le_trans (min_le_left _ _)
    rw [div_le_div_iff₀ (by linarith [ho.1, ho.2]) (by linarith [ho.1])]
    nlinarith [mul_nonneg (sub_nonneg.2 hr.1) (le_trans hlo.le ho.1),
               mul_nonneg (sub_nonneg.2 ho.2) (le_trans hlo.le hr.1)]
  · apply le_trans (min_le_right _ _)
    rw [div_le_div_iff₀ (by linarith [ho.1, ho.2]) (by linarith [ho.2])]
    nlinarith [mul_nonneg (sub_nonneg.2 hr.2) (by linarith [ho.1, ho.2] : (0:ℝ) ≤ 1 - lo),
               mul_nonneg (sub_nonneg.2 ho.1) (by linarith : (0:ℝ) ≤ 1 - hi)]

end Triangular

section DirectionBlindness

theorem task1_ratio_linear {A S : Type*} [NormedAddCommGroup A] [NormedSpace ℝ A]
    [NormedAddCommGroup S] [NormedSpace ℝ S]
    (J : A →L[ℝ] S) (d : A) (hd : J d ≠ 0) (s : ℝ) :
    ‖J (s • d)‖ ^ 2 / ‖J d‖ ^ 2 = s ^ 2 := by
  have hne : ‖J d‖ ≠ 0 := (norm_pos_iff.mpr hd).ne'
  rw [map_smul, norm_smul, Real.norm_eq_abs, mul_pow, sq_abs]
  field_simp

theorem magnitude_ratio_isometry_invariant {S : Type*} [NormedAddCommGroup S] [NormedSpace ℝ S]
    (Q : S →ₗᵢ[ℝ] S) (u v : S) :
    ‖Q u‖ ^ 2 / ‖Q v‖ ^ 2 = ‖u‖ ^ 2 / ‖v‖ ^ 2 := by
  simp [Q.norm_map]

theorem direction_detects_isometry {S : Type*} [NormedAddCommGroup S] [InnerProductSpace ℝ S]
    (v : S) (hv : v ≠ 0) :
    ‖-v‖ = ‖v‖ ∧ (inner (-v) v : ℝ) < 0 ∧ (0 : ℝ) < inner v v := by
  have hp : 0 < ‖v‖ := norm_pos_iff.mpr hv
  refine ⟨norm_neg v, ?_, ?_⟩
  · rw [inner_neg_left, real_inner_self_eq_norm_sq]; nlinarith
  · rw [real_inner_self_eq_norm_sq]; positivity

end DirectionBlindness

section MatchedProbe

variable {S : Type*} [NormedAddCommGroup S] [InnerProductSpace ℝ S]

noncomputable def cosSim (x y : S) : ℝ := (inner x y : ℝ) / (‖x‖ * ‖y‖)

theorem cosSim_eq_inner_normalized (x y : S) :
    cosSim x y = inner (‖x‖⁻¹ • x) (‖y‖⁻¹ • y) := by
  unfold cosSim
  rw [real_inner_smul_left, real_inner_smul_right, div_eq_mul_inv, mul_inv]
  ring

theorem cosSim_smul_pos {c : ℝ} (hc : 0 < c) (x y : S) : cosSim (c • x) y = cosSim x y := by
  unfold cosSim
  rw [real_inner_smul_left, norm_smul, Real.norm_eq_abs, abs_of_pos hc, mul_assoc,
    mul_div_mul_left _ _ hc.ne']

theorem cosSim_neg (x y : S) : cosSim (-x) y = -cosSim x y := by
  unfold cosSim
  rw [inner_neg_left, norm_neg, neg_div]

theorem norm_normalized_le (x : S) : ‖‖x‖⁻¹ • x‖ ≤ 1 := by
  rw [norm_smul, norm_inv, norm_norm]
  rcases eq_or_ne ‖x‖ 0 with h | h
  · simp [h]
  · rw [inv_mul_cancel₀ h]

theorem normalized_sub_le (x y : S) (hy : y ≠ 0) :
    ‖‖x‖⁻¹ • x - ‖y‖⁻¹ • y‖ ≤ 2 * ‖x - y‖ / ‖y‖ := by
  have hyp : 0 < ‖y‖ := norm_pos_iff.mpr hy
  have split : ‖x‖⁻¹ • x - ‖y‖⁻¹ • y = ‖y‖⁻¹ • (x - y) + (‖x‖⁻¹ - ‖y‖⁻¹) • x := by
    rw [smul_sub, sub_smul]; abel
  have h1 : ‖‖y‖⁻¹ • (x - y)‖ = ‖x - y‖ / ‖y‖ := by
    rw [norm_smul, norm_inv, norm_norm, inv_mul_eq_div]
  have h2 : ‖(‖x‖⁻¹ - ‖y‖⁻¹) • x‖ ≤ ‖x - y‖ / ‖y‖ := by
    rcases eq_or_ne x 0 with hx | hx
    · subst hx
      simp only [smul_zero, norm_zero]
      positivity
    · have hxp : 0 < ‖x‖ := norm_pos_iff.mpr hx
      have key : (‖x‖⁻¹ - ‖y‖⁻¹) * ‖x‖ = (‖y‖ - ‖x‖) / ‖y‖ := by
        field_simp
        ring
      calc ‖(‖x‖⁻¹ - ‖y‖⁻¹) • x‖ = |(‖x‖⁻¹ - ‖y‖⁻¹) * ‖x‖| := by
            rw [norm_smul, Real.norm_eq_abs, abs_mul, abs_of_pos hxp]
        _ = |‖y‖ - ‖x‖| / ‖y‖ := by rw [key, abs_div, abs_of_pos hyp]
        _ ≤ ‖x - y‖ / ‖y‖ := by
            apply div_le_div_of_nonneg_right _ hyp.le
            rw [abs_sub_comm]
            exact abs_norm_sub_norm_le x y
  rw [split]
  calc ‖‖y‖⁻¹ • (x - y) + (‖x‖⁻¹ - ‖y‖⁻¹) • x‖
      ≤ ‖‖y‖⁻¹ • (x - y)‖ + ‖(‖x‖⁻¹ - ‖y‖⁻¹) • x‖ := norm_add_le _ _
    _ ≤ ‖x - y‖ / ‖y‖ + ‖x - y‖ / ‖y‖ := by rw [h1]; linarith
    _ = 2 * ‖x - y‖ / ‖y‖ := by ring

theorem cosSim_sub_le (r x y : S) :
    |cosSim r x - cosSim r y| ≤ ‖‖x‖⁻¹ • x - ‖y‖⁻¹ • y‖ := by
  rw [cosSim_eq_inner_normalized, cosSim_eq_inner_normalized, ← inner_sub_right]
  exact (abs_real_inner_le_norm _ _).trans
    (mul_le_of_le_one_left (norm_nonneg _) (norm_normalized_le r))

theorem normalized_dist_sq (x y : S) (hx : x ≠ 0) (hy : y ≠ 0) :
    ‖‖x‖⁻¹ • x - ‖y‖⁻¹ • y‖ ^ 2 = 2 - 2 * cosSim x y := by
  have ux : ‖‖x‖⁻¹ • x‖ = 1 := by
    rw [norm_smul, norm_inv, norm_norm, inv_mul_cancel₀ (norm_ne_zero_iff.mpr hx)]
  have uy : ‖‖y‖⁻¹ • y‖ = 1 := by
    rw [norm_smul, norm_inv, norm_norm, inv_mul_cancel₀ (norm_ne_zero_iff.mpr hy)]
  rw [norm_sub_sq_real, ux, uy, ← cosSim_eq_inner_normalized]
  ring

theorem cosSim_perturb (r v b : S) (hv : v ≠ 0) :
    |cosSim r (v + b) - cosSim r v| ≤ 2 * ‖b‖ / ‖v‖ := by
  rw [cosSim_eq_inner_normalized, cosSim_eq_inner_normalized, ← inner_sub_right]
  calc |(inner (‖r‖⁻¹ • r) (‖v + b‖⁻¹ • (v + b) - ‖v‖⁻¹ • v) : ℝ)|
      ≤ ‖‖r‖⁻¹ • r‖ * ‖‖v + b‖⁻¹ • (v + b) - ‖v‖⁻¹ • v‖ := abs_real_inner_le_norm _ _
    _ ≤ 1 * (2 * ‖(v + b) - v‖ / ‖v‖) :=
        mul_le_mul (norm_normalized_le r) (normalized_sub_le _ _ hv) (norm_nonneg _) zero_le_one
    _ = 2 * ‖b‖ / ‖v‖ := by rw [one_mul, add_sub_cancel_left]

theorem matched_gap_le {ι : Type*} (s : Finset ι) (r v b : ι → S) (hv : ∀ i ∈ s, v i ≠ 0) :
    |∑ i ∈ s, cosSim (r i) (v i + b i) - ∑ i ∈ s, cosSim (r i) (v i)| ≤
      ∑ i ∈ s, 2 * ‖b i‖ / ‖v i‖ := by
  rw [← Finset.sum_sub_distrib]
  exact (Finset.abs_sum_le_sum_abs _ _).trans
    (Finset.sum_le_sum fun i hi => cosSim_perturb _ _ _ (hv i hi))

end MatchedProbe

end WamProofs
