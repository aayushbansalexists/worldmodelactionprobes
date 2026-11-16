import PaperResults

namespace WamProofs

section NoisyReference

variable {S : Type*} [NormedAddCommGroup S] [InnerProductSpace ℝ S]

theorem cosSim_add_noise (r t n : S) (h : (inner r n : ℝ) = 0) :
    cosSim r (t + n) = cosSim r t * (‖t‖ / ‖t + n‖) := by
  unfold cosSim
  rw [inner_add_right, h, add_zero]
  by_cases ht : t = 0
  · simp [ht]
  · have hn : ‖t‖ ≠ 0 := norm_ne_zero_iff.mpr ht
    rw [div_mul_div_comm]
    have hd : ‖r‖ * ‖t‖ * ‖t + n‖ = (‖r‖ * ‖t + n‖) * ‖t‖ := by ring
    rw [hd, mul_div_mul_right _ _ hn]

theorem norm_add_sq_of_inner_zero (t n : S) (h : (inner t n : ℝ) = 0) :
    ‖t + n‖ ^ 2 = ‖t‖ ^ 2 + ‖n‖ ^ 2 := by
  rw [norm_add_sq_real, h]
  ring

theorem noise_factor_sq (t n : S) (h : (inner t n : ℝ) = 0) (hne : t + n ≠ 0) :
    (‖t‖ / ‖t + n‖) ^ 2 = 1 - ‖n‖ ^ 2 / ‖t + n‖ ^ 2 := by
  have hpos : ‖t + n‖ ^ 2 ≠ 0 := pow_ne_zero 2 (norm_ne_zero_iff.mpr hne)
  rw [div_pow, eq_sub_iff_add_eq, ← add_div, ← norm_add_sq_of_inner_zero t n h, div_self hpos]

theorem cosSim_noisy_reference (r t n : S) (hr : (inner r n : ℝ) = 0) (ht : (inner t n : ℝ) = 0)
    (hne : t + n ≠ 0) :
    cosSim r (t + n) = cosSim r t * Real.sqrt (1 - ‖n‖ ^ 2 / ‖t + n‖ ^ 2) := by
  rw [cosSim_add_noise r t n hr, ← noise_factor_sq t n ht hne, Real.sqrt_sq (by positivity)]

end NoisyReference

theorem resamples_of_six_objectives : Nat.choose (6 + 6 - 1) 6 = 462 := by decide

end WamProofs
