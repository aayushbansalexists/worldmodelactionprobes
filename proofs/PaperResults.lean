import WamProofs

open scoped BigOperators
open Finset MeasureTheory

namespace WamProofs

section Transforms
variable {Δ S : Type*} [AddCommGroup Δ] [NormedAddCommGroup S] [InnerProductSpace ℝ S]

def response (f : Δ → S) (A δ : Δ) : S := f (A + δ) - f A

def transformed (f : Δ → S) (A : Δ) (T : S →ₗ[ℝ] S) (a : Δ) : S := f A + T (response f A (a - A))

theorem transformed_at_logged (f : Δ → S) (A : Δ) (T : S →ₗ[ℝ] S) : transformed f A T A = f A := by
  simp [transformed, response]

theorem response_transformed (f : Δ → S) (A : Δ) (T : S →ₗ[ℝ] S) (δ : Δ) :
    response (transformed f A T) A δ = T (response f A δ) := by
  simp [response, transformed]

theorem prop1_size_probe_invariant {ι : Type*} (f : ι → Δ → S) (A : ι → Δ) (Q : S →ₗᵢ[ℝ] S)
    (P : (ι → Δ → ℝ) → ℝ) :
    P (fun i δ => ‖response (transformed (f i) (A i) Q.toLinearMap) (A i) δ‖)
      = P (fun i δ => ‖response (f i) (A i) δ‖) := by
  congr 1
  funext i δ
  rw [response_transformed]
  exact Q.norm_map _

theorem prop1_reflection_direction {ι Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
    (f : ι → Δ → S) (A : ι → Δ) (it : Ω → ι) (dl : Ω → Δ) (rstar : Ω → S) :
    ∫ ω, cosSim (response (transformed (f (it ω)) (A (it ω)) (-LinearMap.id)) (A (it ω)) (dl ω))
        (rstar ω) ∂μ
      = -∫ ω, cosSim (response (f (it ω)) (A (it ω)) (dl ω)) (rstar ω) ∂μ := by
  simp only [response_transformed, LinearMap.neg_apply, LinearMap.id_apply, cosSim_neg]
  exact integral_neg _

end Transforms

section Retrieval
variable {S : Type*} [NormedAddCommGroup S] [InnerProductSpace ℝ S]

theorem prop3_identity (p0 p1 Y : S) :
    ‖p1 - Y‖ ^ 2 - ‖p0 - Y‖ ^ 2 = 2 * (inner (p0 - Y) (p1 - p0) : ℝ) + ‖p1 - p0‖ ^ 2 := by
  have h : p1 - Y = (p0 - Y) + (p1 - p0) := by abel
  rw [h, norm_add_sq_real]
  ring

theorem prop3_scaled (e r : S) (c : ℝ) :
    ‖e + c • r‖ ^ 2 - ‖e‖ ^ 2 = 2 * c * (inner e r : ℝ) + c ^ 2 * ‖r‖ ^ 2 := by
  rw [norm_add_sq_real, real_inner_smul_right, norm_smul, mul_pow, Real.norm_eq_abs, sq_abs]
  ring

lemma prop3_pos_of_large (e r : S) (hr : r ≠ 0) (c : ℝ)
    (hc : 2 * |(inner e r : ℝ)| / ‖r‖ ^ 2 < |c|) :
    0 < 2 * c * (inner e r : ℝ) + c ^ 2 * ‖r‖ ^ 2 := by
  have hn : 0 < ‖r‖ ^ 2 := by have := norm_pos_iff.mpr hr; positivity
  have h1 : 2 * |(inner e r : ℝ)| < |c| * ‖r‖ ^ 2 := by
    have := mul_lt_mul_of_pos_right hc hn
    rwa [div_mul_cancel₀ _ (ne_of_gt hn)] at this
  have hcpos : 0 < |c| := lt_of_le_of_lt (by positivity) hc
  have h2 : -(c * (inner e r : ℝ)) ≤ |c| * |(inner e r : ℝ)| := by
    rw [← abs_mul]; exact neg_le_abs _
  have h3 : c ^ 2 = |c| * |c| := by rw [← sq, sq_abs]
  have h4 : |c| * (2 * |(inner e r : ℝ)|) < |c| * (|c| * ‖r‖ ^ 2) := mul_lt_mul_of_pos_left h1 hcpos
  rw [h3]
  nlinarith [h2, h4]

theorem prop3_large_response {ι : Type*} (s : Finset ι) (e : S) (r : ι → S)
    (hr : ∀ i ∈ s, r i ≠ 0) :
    ∃ C : ℝ, ∀ c : ℝ, C < |c| → ∀ i ∈ s, 0 < ‖e + c • r i‖ ^ 2 - ‖e‖ ^ 2 := by
  refine ⟨∑ i ∈ s, 2 * |(inner e (r i) : ℝ)| / ‖r i‖ ^ 2, fun c hc i hi => ?_⟩
  rw [prop3_scaled]
  apply prop3_pos_of_large e (r i) (hr i hi) c
  refine lt_of_le_of_lt ?_ hc
  exact Finset.single_le_sum (f := fun j => 2 * |(inner e (r j) : ℝ)| / ‖r j‖ ^ 2)
    (fun j _ => by positivity) hi

end Retrieval

section OffSupportFull

theorem prop4_response {H A S : Type*} [AddCommGroup A] [Module ℝ A] [AddCommGroup S] [Module ℝ S]
    (f : H → A → S) (B : H → A →ₗ[ℝ] S) (π : H → A) (h : H) (a δ : A) :
    (f h (a + δ) + B h (a + δ - π h)) - (f h a + B h (a - π h)) = (f h (a + δ) - f h a) + B h δ := by
  rw [show a + δ - π h = (a - π h) + δ by abel, map_add]
  abel

theorem cor5_data_grounded {H A S : Type*} [AddCommGroup A] [Module ℝ A] [AddCommGroup S]
    [Module ℝ S] (f : H → A → S) (B : H → A →ₗ[ℝ] S) (π : H → A) (P : (H → S) → ℝ) :
    P (fun h => f h (π h) + B h (π h - π h)) = P (fun h => f h (π h)) := by
  simp

variable {S : Type*} [NormedAddCommGroup S] [InnerProductSpace ℝ S]

lemma cosSim_self (v : S) (hv : v ≠ 0) : cosSim v v = 1 := by
  have h : ‖v‖ ≠ 0 := norm_ne_zero_iff.mpr hv
  unfold cosSim
  rw [real_inner_self_eq_norm_sq, sq]
  exact div_self (mul_ne_zero h h)

lemma cosSim_neg_right (x y : S) : cosSim x (-y) = -cosSim x y := by
  unfold cosSim
  rw [inner_neg_right, norm_neg, neg_div]

lemma cosSim_continuousAt (v : S) (hv : v ≠ 0) : ContinuousAt (fun x => cosSim x v) v := by
  unfold cosSim
  apply ContinuousAt.div
  · exact (continuous_id.inner continuous_const).continuousAt
  · exact (continuous_norm.mul continuous_const).continuousAt
  · have h : ‖v‖ ≠ 0 := norm_ne_zero_iff.mpr hv
    exact mul_ne_zero h h

lemma tendsto_cosSim_large (r v : S) (hv : v ≠ 0) :
    Filter.Tendsto (fun c : ℝ => cosSim (r + c • v) v) Filter.atTop (nhds 1) := by
  have h1 : Filter.Tendsto (fun c : ℝ => c⁻¹ • r + v) Filter.atTop (nhds v) := by
    have := ((tendsto_inv_atTop_zero : Filter.Tendsto (fun c : ℝ => c⁻¹) Filter.atTop (nhds 0)).smul_const r).add_const v
    simpa using this
  have h2 := (cosSim_continuousAt v hv).tendsto.comp h1
  rw [cosSim_self v hv] at h2
  refine Filter.Tendsto.congr' ?_ h2
  filter_upwards [Filter.eventually_gt_atTop (0 : ℝ)] with c hc
  simp only [Function.comp]
  have e : r + c • v = c • (c⁻¹ • r + v) := by
    rw [smul_add, smul_smul, mul_inv_cancel₀ (ne_of_gt hc), one_smul]
  rw [e, cosSim_smul_pos hc]

theorem cor5_direction_limits {ι : Type*} (s : Finset ι) (hs : s.Nonempty) (r v : ι → S)
    (hv : ∀ i ∈ s, v i ≠ 0) :
    Filter.Tendsto (fun c : ℝ => (1 / (s.card : ℝ)) * ∑ i ∈ s, cosSim (r i + c • v i) (v i))
        Filter.atTop (nhds 1) ∧
      Filter.Tendsto (fun c : ℝ => (1 / (s.card : ℝ)) * ∑ i ∈ s, cosSim (r i + c • (-v i)) (v i))
        Filter.atTop (nhds (-1)) := by
  have hcard : (s.card : ℝ) ≠ 0 := by exact_mod_cast (Finset.card_pos.mpr hs).ne'
  constructor
  · have hsum : Filter.Tendsto (fun c : ℝ => ∑ i ∈ s, cosSim (r i + c • v i) (v i)) Filter.atTop
        (nhds (∑ _i ∈ s, (1 : ℝ))) :=
      tendsto_finset_sum s (fun i hi => tendsto_cosSim_large (r i) (v i) (hv i hi))
    have := hsum.const_mul (1 / (s.card : ℝ))
    simpa [hcard] using this
  · have hsum : Filter.Tendsto (fun c : ℝ => ∑ i ∈ s, cosSim (r i + c • (-v i)) (v i)) Filter.atTop
        (nhds (∑ _i ∈ s, (-1 : ℝ))) := by
      refine tendsto_finset_sum s (fun i hi => ?_)
      have hn : -v i ≠ 0 := neg_ne_zero.mpr (hv i hi)
      have := (tendsto_cosSim_large (r i) (-v i) hn).neg
      refine Filter.Tendsto.congr' ?_ this
      filter_upwards with c
      rw [← neg_neg (v i), cosSim_neg_right, neg_neg, neg_neg]
    have := hsum.const_mul (1 / (s.card : ℝ))
    simpa [hcard] using this

end OffSupportFull

section Matched
variable {S : Type*} [NormedAddCommGroup S] [InnerProductSpace ℝ S]

theorem prop6_scaled_swap {Δ : Type*} [AddCommGroup Δ] (f : Δ → S) (A δ : Δ) (c : ℝ) :
    response (transformed f A (c • LinearMap.id)) A δ = c • response f A δ := by
  rw [response_transformed]; simp

theorem prop6_i {ι : Type*} (s : Finset ι) (r e : ι → S) (c : ℝ) (hc : 0 < c) :
    ∑ i ∈ s, cosSim (c • r i) (e i) = ∑ i ∈ s, cosSim (r i) (e i) :=
  Finset.sum_congr rfl (fun i _ => cosSim_smul_pos hc (r i) (e i))

theorem prop6_ii {ι : Type*} (s : Finset ι) (r e : ι → S) :
    ∑ i ∈ s, cosSim (-r i) (e i) = -∑ i ∈ s, cosSim (r i) (e i) := by
  rw [← Finset.sum_neg_distrib]
  exact Finset.sum_congr rfl (fun i _ => cosSim_neg (r i) (e i))

theorem prop6_iii (r e v : S) (he : e ≠ 0) (hv : v ≠ 0) :
    |cosSim r e - cosSim r v| ≤ Real.sqrt (2 - 2 * cosSim e v) := by
  calc |cosSim r e - cosSim r v| ≤ ‖‖e‖⁻¹ • e - ‖v‖⁻¹ • v‖ := cosSim_sub_le r e v
    _ = Real.sqrt (‖‖e‖⁻¹ • e - ‖v‖⁻¹ • v‖ ^ 2) := (Real.sqrt_sq (norm_nonneg _)).symm
    _ = Real.sqrt (2 - 2 * cosSim e v) := by rw [normalized_dist_sq e v he hv]

theorem prop6_example (r e v : S) (he : e ≠ 0) (hv : v ≠ 0) (h : 0.9 ≤ cosSim e v) :
    |cosSim r e - cosSim r v| ≤ 0.45 := by
  refine le_trans (prop6_iii r e v he hv) ?_
  have h1 : Real.sqrt (2 - 2 * cosSim e v) ≤ Real.sqrt (0.45 ^ 2) :=
    Real.sqrt_le_sqrt (by nlinarith)
  rwa [Real.sqrt_sq (by norm_num)] at h1

end Matched

section Attenuation

variable {n : ℕ}
local notation "E" => EuclideanSpace ℝ (Fin n)

noncomputable def unitVec (x : E) : E := ‖x‖⁻¹ • x

lemma measurable_unitVec : Measurable (unitVec (n := n)) :=
  (measurable_norm.inv).smul measurable_id

lemma norm_unitVec_le (x : E) : ‖unitVec x‖ ≤ 1 := norm_normalized_le x

lemma integrable_unitVec_shift (μ : Measure E) [IsProbabilityMeasure μ] (v : E) :
    Integrable (fun ξ => unitVec (v + ξ)) μ := by
  refine Integrable.mono' (integrable_const (1 : ℝ)) ?_ (Filter.Eventually.of_forall fun ξ => ?_)
  · exact (measurable_unitVec.comp (measurable_const.add measurable_id)).aestronglyMeasurable
  · exact norm_unitVec_le _

lemma unitVec_isometry (L : E ≃ₗᵢ[ℝ] E) (x : E) : unitVec (L x) = L (unitVec x) := by
  simp [unitVec, L.norm_map, _root_.map_smul]

lemma mono_ratio {p q dp dq c : ℝ} (hc : 0 ≤ c) (hp : 0 ≤ p) (hpq : p ≤ q) (hdp : 0 ≤ dp)
    (hdq : 0 ≤ dq) (hdp2 : dp ^ 2 = p ^ 2 + c) (hdq2 : dq ^ 2 = q ^ 2 + c) : p / dp ≤ q / dq := by
  rcases eq_or_lt_of_le hp with h0 | hppos
  · rw [← h0, zero_div]; exact div_nonneg (le_trans hp hpq) hdq
  · have hqpos : 0 < q := lt_of_lt_of_le hppos hpq
    have hdpp : 0 < dp := by
      rcases eq_or_lt_of_le hdp with h | h
      · rw [← h] at hdp2; nlinarith
      · exact h
    have hdqp : 0 < dq := by
      rcases eq_or_lt_of_le hdq with h | h
      · rw [← h] at hdq2; nlinarith
      · exact h
    rw [div_le_div_iff₀ hdpp hdqp]
    have hsq : (p * dq) ^ 2 ≤ (q * dp) ^ 2 := by
      rw [mul_pow, mul_pow, hdp2, hdq2]; nlinarith [sq_nonneg p, sq_nonneg q, mul_le_mul hpq hpq hp (le_of_lt hqpos)]
    nlinarith [sq_nonneg (p * dq - q * dp), sq_nonneg (p * dq + q * dp), mul_pos hppos hdqp, mul_pos hqpos hdpp]

lemma pair_nonneg {a t c d1 d2 : ℝ} (ha : 0 < a) (hc : 0 ≤ c) (hd1 : 0 ≤ d1) (hd2 : 0 ≤ d2)
    (h1 : d1 ^ 2 = (a + t) ^ 2 + c) (h2 : d2 ^ 2 = (a - t) ^ 2 + c) :
    0 ≤ (a + t) / d1 + (a - t) / d2 := by
  rcases le_total 0 t with ht | ht
  · rcases le_total 0 (a - t) with hat | hat
    · exact add_nonneg (div_nonneg (by linarith) hd1) (div_nonneg hat hd2)
    · have hm := mono_ratio (p := t - a) (q := a + t) hc (by linarith) (by linarith) hd2 hd1
        (by rw [h2]; ring) h1
      have : (a - t) / d2 = -((t - a) / d2) := by rw [← neg_div]; ring_nf
      linarith
  · rcases le_total 0 (a + t) with hat | hat
    · exact add_nonneg (div_nonneg hat hd1) (div_nonneg (by linarith) hd2)
    · have hm := mono_ratio (p := -(a + t)) (q := a - t) hc (by linarith) (by linarith) hd1 hd2
        (by rw [h1]; ring) h2
      have : (a + t) / d1 = -((-(a + t)) / d1) := by rw [neg_div, neg_neg]
      linarith

theorem prop7_item (μ : Measure E) [IsProbabilityMeasure μ] (v : E) (hv : v ≠ 0)
    (hP : MeasurePreserving (reflection (Submodule.span ℝ {v})) μ μ)
    (hR : MeasurePreserving (reflection (Submodule.span ℝ {v})ᗮ) μ μ) :
    ∃ κ : ℝ, 0 ≤ κ ∧ κ ≤ 1 ∧ ∀ r : E, ∫ ξ, cosSim r (v + ξ) ∂μ = κ * cosSim r v := by
  set K := Submodule.span ℝ {v}
  set P := reflection K
  set R := reflection Kᗮ
  have hvK : v ∈ K := Submodule.mem_span_singleton_self v
  have hPv : P v = v := reflection_mem_subspace_eq_self hvK
  have hRv : R v = -v := reflection_orthogonalComplement_singleton_eq_neg v
  have hint := integrable_unitVec_shift μ v
  set m := ∫ ξ, unitVec (v + ξ) ∂μ with hm

  have hPm : P m = m := by
    have h1 : ∫ ξ, unitVec (v + P ξ) ∂μ = m :=
      hP.integral_comp P.toHomeomorph.measurableEmbedding (fun ξ => unitVec (v + ξ))
    have h2 : ∫ ξ, unitVec (v + P ξ) ∂μ = P m := by
      have : (fun ξ => unitVec (v + P ξ)) = fun ξ => P.toLinearIsometry (unitVec (v + ξ)) := by
        funext ξ
        have e : v + P ξ = P (v + ξ) := by rw [map_add, hPv]
        rw [e, unitVec_isometry]; rfl
      rw [this, LinearIsometry.integral_comp_comm]; rfl
    rw [← h2, h1]
  have hmK : m ∈ K := (reflection_eq_self_iff m).mp hPm
  obtain ⟨b, hb⟩ := Submodule.mem_span_singleton.mp hmK
  have hnv : 0 < ‖v‖ := norm_pos_iff.mpr hv

  have hcos : ∀ r : E, ∫ ξ, cosSim r (v + ξ) ∂μ = (b * ‖v‖) * cosSim r v := by
    intro r
    have h1 : ∀ ξ, cosSim r (v + ξ) = (inner (‖r‖⁻¹ • r) (unitVec (v + ξ)) : ℝ) := fun ξ =>
      cosSim_eq_inner_normalized r (v + ξ)
    simp_rw [h1]
    have hvn : ‖v‖ ≠ 0 := hnv.ne'
    rw [integral_inner hint, ← hm, ← hb, cosSim_eq_inner_normalized r v]
    simp only [real_inner_smul_right]
    field_simp
    ring
  refine ⟨b * ‖v‖, ?_, ?_, hcos⟩
  ·
    have hk : ∫ ξ, cosSim v (v + ξ) ∂μ = b * ‖v‖ := by rw [hcos v, cosSim_self v hv, mul_one]
    have hkR : ∫ ξ, cosSim v (v + R ξ) ∂μ = b * ‖v‖ := by
      rw [← hk]; exact hR.integral_comp R.toHomeomorph.measurableEmbedding (fun ξ => cosSim v (v + ξ))
    have hmeas : ∀ w : E, Integrable (fun ξ => cosSim v (w + ξ)) μ := by
      intro w
      refine Integrable.mono' (integrable_const (1 : ℝ)) ?_ (Filter.Eventually.of_forall fun ξ => ?_)
      · have : (fun ξ => cosSim v (w + ξ)) = fun ξ => (inner (‖v‖⁻¹ • v) (unitVec (w + ξ)) : ℝ) := by
          funext ξ; exact cosSim_eq_inner_normalized v (w + ξ)
        rw [this]
        exact (measurable_const.inner (measurable_unitVec.comp (measurable_const.add measurable_id))).aestronglyMeasurable
      · rw [Real.norm_eq_abs]; exact abs_real_inner_div_norm_mul_norm_le_one v (w + ξ)
    have hRint : Integrable (fun ξ => cosSim v (v + R ξ)) μ :=
      (hR.integrable_comp_emb R.toHomeomorph.measurableEmbedding).mpr (hmeas v)
    have hsum : 0 ≤ ∫ ξ, (cosSim v (v + ξ) + cosSim v (v + R ξ)) ∂μ := by
      refine integral_nonneg (fun ξ => ?_)
      have hvn : ‖v‖ ≠ 0 := hnv.ne'
      set a := ‖v‖ with ha
      set t := (inner v ξ : ℝ) with ht
      have hRs : (inner v (R ξ) : ℝ) = -t := by
        have h := R.inner_map_map v (R ξ)
        rw [hRv, reflection_reflection, inner_neg_left] at h
        linarith
      have hRn : ‖R ξ‖ = ‖ξ‖ := R.norm_map ξ
      have hcs : |t| ≤ a * ‖ξ‖ := abs_real_inner_le_norm v ξ
      have hc : 0 ≤ ‖ξ‖ ^ 2 - (t / a) ^ 2 := by
        have h1 : t ^ 2 ≤ (a * ‖ξ‖) ^ 2 := by
          have := mul_self_le_mul_self (abs_nonneg t) hcs
          rw [abs_mul_abs_self] at this; nlinarith
        have h2 : (t / a) ^ 2 ≤ ‖ξ‖ ^ 2 := by
          rw [div_pow, div_le_iff₀ (by positivity)]; nlinarith
        linarith
      have q1 : a + t / a = (a ^ 2 + t) / a := by field_simp; ring
      have q2 : a - t / a = (a ^ 2 + -t) / a := by field_simp; ring
      have e1 : cosSim v (v + ξ) = (a + t / a) / ‖v + ξ‖ := by
        unfold cosSim
        rw [inner_add_right, real_inner_self_eq_norm_sq, q1, div_div]
      have e2 : cosSim v (v + R ξ) = (a - t / a) / ‖v + R ξ‖ := by
        unfold cosSim
        rw [inner_add_right, real_inner_self_eq_norm_sq, hRs, q2, div_div]
      have d1 : ‖v + ξ‖ ^ 2 = (a + t / a) ^ 2 + (‖ξ‖ ^ 2 - (t / a) ^ 2) := by
        rw [norm_add_sq_real, ← ht]
        field_simp
        ring
      have d2 : ‖v + R ξ‖ ^ 2 = (a - t / a) ^ 2 + (‖ξ‖ ^ 2 - (t / a) ^ 2) := by
        rw [norm_add_sq_real, hRs, hRn]
        field_simp
        ring
      rw [e1, e2]
      exact pair_nonneg hnv hc (norm_nonneg _) (norm_nonneg _) d1 d2
    have : ∫ ξ, (cosSim v (v + ξ) + cosSim v (v + R ξ)) ∂μ = 2 * (b * ‖v‖) := by
      rw [integral_add (hmeas v) hRint, hk, hkR]; ring
    linarith
  · have hk : ∫ ξ, cosSim v (v + ξ) ∂μ = b * ‖v‖ := by rw [hcos v, cosSim_self v hv, mul_one]
    have hle : ‖∫ ξ, cosSim v (v + ξ) ∂μ‖ ≤ 1 := by
      have := norm_integral_le_of_norm_le_const (μ := μ) (C := 1)
        (Filter.Eventually.of_forall fun ξ => by
          rw [Real.norm_eq_abs]; exact abs_real_inner_div_norm_mul_norm_le_one v (v + ξ))
      simpa using this
    rw [hk, Real.norm_eq_abs] at hle
    exact le_trans (le_abs_self _) hle

theorem prop7_ordering {ι : Type*} (s : Finset ι) (κ x y : ι → ℝ) (k : ℝ) (hk : 0 < k)
    (hκ : ∀ i ∈ s, κ i = k) :
    ∑ i ∈ s, κ i * x i ≤ ∑ i ∈ s, κ i * y i ↔ ∑ i ∈ s, x i ≤ ∑ i ∈ s, y i := by
  have h1 : ∑ i ∈ s, κ i * x i = k * ∑ i ∈ s, x i := by
    rw [Finset.mul_sum]; exact Finset.sum_congr rfl (fun i hi => by rw [hκ i hi])
  have h2 : ∑ i ∈ s, κ i * y i = k * ∑ i ∈ s, y i := by
    rw [Finset.mul_sum]; exact Finset.sum_congr rfl (fun i hi => by rw [hκ i hi])
  rw [h1, h2]
  exact mul_le_mul_left hk

lemma cosSim_self_smul (v : E) (hv : v ≠ 0) (c : ℝ) (hc : 0 < c) : cosSim v (c • v) = 1 := by
  unfold cosSim
  rw [real_inner_smul_right, real_inner_self_eq_norm_sq, norm_smul, Real.norm_eq_abs, abs_of_pos hc]
  have : ‖v‖ ≠ 0 := norm_ne_zero_iff.mpr hv
  field_simp
  ring

theorem prop7_zero_example (v : E) (hv : v ≠ 0) :
    (1 / 2 : ℝ) * cosSim v (v + (3 : ℝ) • v) + (1 / 2) * cosSim v (v + (-3 : ℝ) • v) = 0 := by
  have e1 : v + (3 : ℝ) • v = (4 : ℝ) • v := by
    rw [show (4 : ℝ) = 1 + 3 by norm_num, add_smul, one_smul]
  have e2 : v + (-3 : ℝ) • v = -((2 : ℝ) • v) := by
    rw [show (-3 : ℝ) = -(1 + 2) by norm_num, neg_smul, add_smul, one_smul]; abel
  rw [e1, e2, cosSim_neg_right, cosSim_self_smul v hv 4 (by norm_num), cosSim_self_smul v hv 2 (by norm_num)]
  ring

theorem prop7_mean {ι Ω : Type*} [MeasurableSpace Ω] (Pr : Measure Ω) [IsProbabilityMeasure Pr]
    (s : Finset ι) (r v : ι → E) (hv : ∀ i, v i ≠ 0) (ξ : ι → Ω → E) (hξ : ∀ i, Measurable (ξ i))
    (hP : ∀ i, MeasurePreserving (reflection (Submodule.span ℝ {v i})) (Pr.map (ξ i)) (Pr.map (ξ i)))
    (hR : ∀ i, MeasurePreserving (reflection (Submodule.span ℝ {v i})ᗮ) (Pr.map (ξ i)) (Pr.map (ξ i))) :
    ∃ κ : ι → ℝ, (∀ i, 0 ≤ κ i ∧ κ i ≤ 1) ∧
      ∫ ω, (1 / (s.card : ℝ)) * ∑ i ∈ s, cosSim (r i) (v i + ξ i ω) ∂Pr
        = (1 / (s.card : ℝ)) * ∑ i ∈ s, κ i * cosSim (r i) (v i) := by
  have hprob : ∀ i, IsProbabilityMeasure (Pr.map (ξ i)) :=
    fun i => isProbabilityMeasure_map (hξ i).aemeasurable
  have hex : ∀ i, ∃ k : ℝ, 0 ≤ k ∧ k ≤ 1 ∧ ∀ q : E, ∫ x, cosSim q (v i + x) ∂(Pr.map (ξ i)) = k * cosSim q (v i) :=
    fun i => by haveI := hprob i; exact prop7_item (Pr.map (ξ i)) (v i) (hv i) (hP i) (hR i)
  choose κ hκ0 hκ1 hκ using hex
  refine ⟨κ, fun i => ⟨hκ0 i, hκ1 i⟩, ?_⟩
  have hform : ∀ i (w : Ω → E), (fun ω => cosSim (r i) (v i + w ω))
      = fun ω => (inner (‖r i‖⁻¹ • r i) (unitVec (v i + w ω)) : ℝ) := by
    intro i w; funext ω; exact cosSim_eq_inner_normalized _ _
  have hint : ∀ i ∈ s, Integrable (fun ω => cosSim (r i) (v i + ξ i ω)) Pr := by
    intro i _
    refine Integrable.mono' (integrable_const (1 : ℝ)) ?_ (Filter.Eventually.of_forall fun ω => ?_)
    · rw [hform i (ξ i)]
      exact (measurable_const.inner (measurable_unitVec.comp (measurable_const.add (hξ i)))).aestronglyMeasurable
    · rw [Real.norm_eq_abs]; exact abs_real_inner_div_norm_mul_norm_le_one _ _
  rw [integral_mul_left, integral_finset_sum s hint]
  congr 1
  refine Finset.sum_congr rfl (fun i _ => ?_)
  have hmeas : AEStronglyMeasurable (fun x => cosSim (r i) (v i + x)) (Pr.map (ξ i)) := by
    have hf : (fun x : E => cosSim (r i) (v i + x))
        = fun x => (inner (‖r i‖⁻¹ • r i) (unitVec (v i + x)) : ℝ) := by
      funext x; exact cosSim_eq_inner_normalized _ _
    rw [hf]
    exact (measurable_const.inner (measurable_unitVec.comp (measurable_const.add measurable_id))).aestronglyMeasurable
  rw [← hκ i (r i)]
  exact (integral_map (hξ i).aemeasurable hmeas).symm

end Attenuation

section Inverse

theorem prop8_affine_step {X A : Type*} [AddCommGroup X] [Module ℝ X] [AddCommGroup A] [Module ℝ A]
    (W1 W2 : X →ₗ[ℝ] A) (b : A) (x y d0 d1 : X) :
    (W1 (x + d0) + W2 (y + d1) + b) - (W1 x + W2 y + b) = W1 d0 + W2 d1 := by
  simp only [map_add]; abel

theorem prop8 {Z Y : Type*} [AddCommGroup Z] [Module ℝ Z] [NormedAddCommGroup Y] [InnerProductSpace ℝ Y]
    (L : Z →ₗ[ℝ] Y) (Δ : Z) (a : Y) (c : ℝ) (hc : 0 < c) :
    cosSim (L (c • Δ)) a = cosSim (L Δ) a ∧ cosSim (L (-Δ)) a = -cosSim (L Δ) a := by
  rw [_root_.map_smul, map_neg]
  exact ⟨cosSim_smul_pos hc _ _, cosSim_neg _ _⟩

theorem prop8_nonlinear :
    cosSim ((2 : ℝ) ^ 2 - 2 - (0 ^ 2 - 0)) (1 : ℝ) = 1 ∧
      cosSim ((0.5 : ℝ) ^ 2 - 0.5 - (0 ^ 2 - 0)) (1 : ℝ) = -1 ∧
      cosSim ((-2 : ℝ) ^ 2 - (-2) - (0 ^ 2 - 0)) (1 : ℝ) = 1 := by
  refine ⟨?_, ?_, ?_⟩ <;> norm_num [cosSim, RCLike.inner_apply, conj_trivial, Real.norm_eq_abs, abs_of_pos, abs_of_neg]

end Inverse

section AppendixD
open Matrix

theorem appD_reflection_cov {m : Type*} [Fintype m] [DecidableEq m] (S : Matrix m m ℝ) :
    (-1 : Matrix m m ℝ) * S * (-1 : Matrix m m ℝ)ᵀ = S := by
  simp

theorem appD_rotation_det {m : Type*} [Fintype m] [DecidableEq m] (S Q : Matrix m m ℝ)
    (hQ : Q * Qᵀ = 1) (a : ℝ) : (1 + a • (Q * S * Qᵀ)).det = (1 + a • S).det := by
  have h : 1 + a • (Q * S * Qᵀ) = Q * (1 + a • S) * Qᵀ := by
    rw [Matrix.mul_add, Matrix.add_mul, Matrix.mul_one, hQ, Matrix.mul_smul, Matrix.smul_mul]
  have hq : Q.det * Q.det = 1 := by
    have := congrArg Matrix.det hQ
    rwa [det_mul, det_transpose, det_one] at this
  rw [h, det_mul, det_mul, det_transpose]
  linear_combination (1 + a • S).det * hq

variable {S : Type*} [NormedAddCommGroup S] [InnerProductSpace ℝ S]

theorem appD_consistency (T : S →ₗᵢ[ℝ] S) (x y v u : S) :
    ‖T x - T y‖ = ‖x - y‖ ∧ ‖T v‖ = ‖v‖ ∧ ‖(-u - u) + -(0 : S)‖ = 2 * ‖u‖ := by
  refine ⟨?_, T.norm_map v, ?_⟩
  · rw [← map_sub]; exact T.norm_map _
  · rw [neg_zero, add_zero, show -u - u = -((2 : ℝ) • u) by rw [two_smul]; abel, norm_neg, norm_smul]
    norm_num

end AppendixD

end WamProofs
