package turboquant

import (
	"math"
	"math/rand/v2"
	"testing"
)

// reconstructVec round-trips a vector through EncodeKeyVector →
// MarshalBinary → UnmarshalBinary → DecodeVector for a given preset. All
// recon tests in this file use this shared helper so the 4-variant sweep
// compares the same encode/marshal path for every preset.
func reconstructVec(t *testing.T, v []float32, preset Preset) []float32 {
	t.Helper()
	ev, err := EncodeKeyVector(v, preset)
	if err != nil {
		t.Fatalf("encode %s: %v", preset.Name, err)
	}
	raw, err := ev.MarshalBinary()
	if err != nil {
		t.Fatalf("marshal %s: %v", preset.Name, err)
	}
	out, _, err := DecodeVector(raw)
	if err != nil {
		t.Fatalf("decode %s: %v", preset.Name, err)
	}
	return out
}

// runReconSweep evaluates a set of presets over `trials` random draws from
// `makeKey` and returns per-preset slices of per-trial dot-product error and
// L2 reconstruction error. The query vector q is fixed across trials so each
// preset is scored on the same (q, v) pairs. Used by the 4-variant gate and
// safety tests below.
func runReconSweep(t *testing.T, trials int, makeKey func() []float32, q []float32, presets []Preset) (dot [][]float64, l2 [][]float64) {
	t.Helper()

	dot = make([][]float64, len(presets))
	l2 = make([][]float64, len(presets))
	for i := range presets {
		dot[i] = make([]float64, trials)
		l2[i] = make([]float64, trials)
	}

	dotProd := func(x, y []float32) float64 {
		var s float64
		for i := range x {
			s += float64(x[i]) * float64(y[i])
		}
		return s
	}
	l2sq := func(u, w []float32) float64 {
		var s float64
		for i := range u {
			d := float64(u[i] - w[i])
			s += d * d
		}
		return s
	}

	for k := 0; k < trials; k++ {
		v := makeKey()
		gold := dotProd(q, v)
		for pi, p := range presets {
			rec := reconstructVec(t, v, p)
			dot[pi][k] = math.Abs(dotProd(q, rec) - gold)
			l2[pi][k] = math.Sqrt(l2sq(v, rec))
		}
	}
	return
}

// meanStd returns mean and sample standard deviation.
func meanStd(x []float64) (mean, std float64) {
	for _, v := range x {
		mean += v
	}
	mean /= float64(len(x))
	for _, v := range x {
		d := v - mean
		std += d * d
	}
	std = math.Sqrt(std / float64(len(x)-1))
	return
}

// makeGaussianQuery returns a deterministic Gaussian query vector.
func makeGaussianQuery(rng *rand.Rand, headDim int) []float32 {
	q := make([]float32, headDim)
	for i := range q {
		q[i] = float32(rng.NormFloat64())
	}
	return q
}

// TestQJLRecoversAsymmetricKey is the four-variant capability gate on the
// Qwen 2 weak spot. On the biased distribution described in the PR body
// (Gaussian coordinates plus a learned per-channel bias with a handful of
// heavy outlier channels) the assertions land three honest constraints:
//
//   1. tq3q must improve on tq3 by at least 15% — this is the original
//      QJL claim and the main reason tq3q exists as a preset.
//
//   2. The stacked combination tq3qa must not regress on the better of
//      the two single-mechanism presets (tq3a, tq3q) by more than 5%.
//      "At least as good as the better single mechanism, minus a small
//      tolerance" is the right bar: if one mechanism already captured
//      the error the other was correcting, adding the other may be a
//      wash rather than a strict win.
//
//   3. tq3a must not regress on tq3 — centred-asymmetric primary is
//      allowed to be a wash on distributions where the per-block mean
//      of rotated coordinates is close to zero (which happens on this
//      synthetic distribution: the learned bias is zero-mean per-channel,
//      with only 8 heavy outlier channels). Regressing would mean the
//      encoder is breaking; a wash is expected.
//
// The test logs a 4-column table so a reviewer running `go test -v` sees
// the four values and relative deltas regardless of whether the bars are
// met tightly or with slack.
func TestQJLRecoversAsymmetricKey(t *testing.T) {
	const (
		headDim = 128
		trials  = 128
	)

	rng := rand.New(rand.NewPCG(0x9e3779b97f4a7c15, 0xbf58476d1ce4e5b9))

	makeKey := func() []float32 {
		bias := make([]float32, headDim)
		for i := range bias {
			bias[i] = 0.25 * float32(rng.NormFloat64())
		}
		// Heavy outlier channels — the signature of a learned K bias.
		for k := 0; k < 8; k++ {
			idx := rng.IntN(headDim)
			bias[idx] += float32(math.Copysign(2.0+rng.Float64(), rng.NormFloat64()))
		}
		v := make([]float32, headDim)
		for i := range v {
			v[i] = float32(rng.NormFloat64()) + bias[i]
		}
		return v
	}

	q := makeGaussianQuery(rng, headDim)

	presets := []Preset{PresetTQ3, PresetTQ3A, PresetTQ3Q, PresetTQ3QA}
	dot, l2 := runReconSweep(t, trials, makeKey, q, presets)

	type row struct {
		name string
		dot  float64
		l2   float64
	}
	rows := make([]row, len(presets))
	for i, p := range presets {
		dm, _ := meanStd(dot[i])
		lm, _ := meanStd(l2[i])
		rows[i] = row{name: p.Name, dot: dm, l2: lm}
	}

	// Baseline is tq3 (first in the slice) — asserted below. Print the table
	// alongside relative deltas so a reviewer running `go test -v` sees the
	// four numbers side-by-side without reading the assertions.
	t.Logf("asymmetric-K (n=%d):", trials)
	t.Logf("  %-7s %-10s %-10s %-10s", "preset", "dot-err", "L2-err", "Δvs tq3")
	for _, r := range rows {
		rel := (r.dot - rows[0].dot) / rows[0].dot * 100
		t.Logf("  %-7s %-10.4f %-10.4f %+.1f%%", r.name, r.dot, r.l2, rel)
	}

	tq3 := rows[0].dot
	tq3a := rows[1].dot
	tq3q := rows[2].dot
	tq3qa := rows[3].dot

	// Assertion 1: tq3q must improve tq3 by ≥15%. This is the original QJL
	// claim and what the preset primarily exists to enable.
	tq3qImp := (tq3 - tq3q) / tq3
	if tq3qImp < 0.15 {
		t.Errorf("tq3q improvement over tq3 was only %.1f%% — below the 15%% floor", 100*tq3qImp)
	}

	// Assertion 2: tq3qa must be within 5% of the better of tq3a / tq3q.
	// Stacking should not regress; a wash (if one mechanism already absorbed
	// the other's error) is acceptable.
	best := math.Min(tq3a, tq3q)
	if tq3qa > best*1.05 {
		t.Errorf("tq3qa (%.4f) regresses on the better of tq3a / tq3q (%.4f) by more than 5%%", tq3qa, best)
	}

	// Assertion 3: tq3a must not regress tq3. Centred-asymmetric is allowed
	// to be a wash on distributions whose per-block rotated-mean is near zero
	// (which is the case here: the bias in makeKey is per-channel zero-mean
	// plus 8 heavy outlier channels, so rotated-mean is small). Small
	// per-block scale changes from RMS-around-mean vs raw RMS can go either
	// way; require only that the preset doesn't materially break things.
	if tq3a > tq3*1.05 {
		t.Errorf("tq3a (%.4f) regresses on tq3 (%.4f) by more than 5%%", tq3a, tq3)
	}
}

// TestQJLRecoversAsymmetricKey2Bit mirrors TestQJLRecoversAsymmetricKey at
// 2-bit primary quantization. The mechanisms (centred-asymmetric primary,
// QJL residual) are bit-width-agnostic; this test confirms they continue to
// stack and improve on the symmetric 2-bit baseline. The expected result is
// a *larger* relative improvement than at 3 bits — symmetric tq2 has only 4
// codebook centroids, so wasting one of them on an empty distribution half
// (the failure mode centring corrects) costs ~25% of the representational
// budget vs ~12.5% at tq3.
func TestQJLRecoversAsymmetricKey2Bit(t *testing.T) {
	const (
		headDim = 128
		trials  = 128
	)

	rng := rand.New(rand.NewPCG(0x9e3779b97f4a7c15, 0xbf58476d1ce4e5b9))

	makeKey := func() []float32 {
		bias := make([]float32, headDim)
		for i := range bias {
			bias[i] = 0.25 * float32(rng.NormFloat64())
		}
		for k := 0; k < 8; k++ {
			idx := rng.IntN(headDim)
			bias[idx] += float32(math.Copysign(2.0+rng.Float64(), rng.NormFloat64()))
		}
		v := make([]float32, headDim)
		for i := range v {
			v[i] = float32(rng.NormFloat64()) + bias[i]
		}
		return v
	}

	q := makeGaussianQuery(rng, headDim)

	presets := []Preset{PresetTQ2, PresetTQ2A, PresetTQ2Q, PresetTQ2QA}
	dot, l2 := runReconSweep(t, trials, makeKey, q, presets)

	type row struct {
		name string
		dot  float64
		l2   float64
	}
	rows := make([]row, len(presets))
	for i, p := range presets {
		dm, _ := meanStd(dot[i])
		lm, _ := meanStd(l2[i])
		rows[i] = row{name: p.Name, dot: dm, l2: lm}
	}

	t.Logf("asymmetric-K (n=%d, 2-bit primary):", trials)
	t.Logf("  %-7s %-10s %-10s %-10s", "preset", "dot-err", "L2-err", "Δvs tq2")
	for _, r := range rows {
		rel := (r.dot - rows[0].dot) / rows[0].dot * 100
		t.Logf("  %-7s %-10.4f %-10.4f %+.1f%%", r.name, r.dot, r.l2, rel)
	}

	tq2 := rows[0].dot
	tq2a := rows[1].dot
	tq2q := rows[2].dot
	tq2qa := rows[3].dot

	// At 2 bits, tq2q's QJL residual must improve tq2 by at least 15% — the
	// same floor used at 3 bits. We expect more, but ≥15% is what we'll
	// commit to as the gate.
	tq2qImp := (tq2 - tq2q) / tq2
	if tq2qImp < 0.15 {
		t.Errorf("tq2q improvement over tq2 was only %.1f%% — below the 15%% floor", 100*tq2qImp)
	}

	// Stacked combination must not regress on the better of the two single-
	// mechanism presets, with 5% tolerance for noise.
	best := math.Min(tq2a, tq2q)
	if tq2qa > best*1.05 {
		t.Errorf("tq2qa (%.4f) regresses on the better of tq2a / tq2q (%.4f) by more than 5%%", tq2qa, best)
	}

	// tq2a may be a wash on this distribution (same reasoning as 3-bit) but
	// must not regress materially.
	if tq2a > tq2*1.05 {
		t.Errorf("tq2a (%.4f) regresses on tq2 (%.4f) by more than 5%%", tq2a, tq2)
	}
}

// TestQJLDoesNotRegressSymmetricKey is the safety test: on the well-behaved
// symmetric-Gaussian distribution where the shipped tq3 preset was tuned to
// be near-f16, none of the new presets (tq3a, tq3q, tq3qa) may regress dot-
// product reconstruction error by more than 2× tq3. A small degradation is
// tolerable — outlier split carves typical values into a lower-resolution
// sub-block, centring shifts the quantisation mid-range — but a catastrophic
// one would mean the presets are unsafe as drop-ins for well-behaved models
// like llama3.2 / gemma3 / qwen3-coder.
func TestQJLDoesNotRegressSymmetricKey(t *testing.T) {
	const (
		headDim = 128
		trials  = 128
	)

	rng := rand.New(rand.NewPCG(0xbf58476d1ce4e5b9, 0x94d049bb133111eb))

	// Pure Gaussian coordinates — no learned bias, no heavy channels. This
	// is the "easy" distribution that the existing tq3 preset was tuned
	// against on llama / gemma / qwen3-coder in the PR body's measured
	// benchmark matrix.
	makeKey := func() []float32 {
		v := make([]float32, headDim)
		for i := range v {
			v[i] = float32(rng.NormFloat64())
		}
		return v
	}

	q := makeGaussianQuery(rng, headDim)

	presets := []Preset{PresetTQ3, PresetTQ3A, PresetTQ3Q, PresetTQ3QA}
	dot, l2 := runReconSweep(t, trials, makeKey, q, presets)

	rows := make([]float64, len(presets))
	for i := range presets {
		rows[i], _ = meanStd(dot[i])
	}

	t.Logf("symmetric-K (n=%d):", trials)
	t.Logf("  %-7s %-10s %-10s %-10s", "preset", "dot-err", "L2-err", "Δvs tq3")
	for i, p := range presets {
		lm, _ := meanStd(l2[i])
		rel := (rows[i] - rows[0]) / rows[0] * 100
		t.Logf("  %-7s %-10.4f %-10.4f %+.1f%%", p.Name, rows[i], lm, rel)
	}

	const regressionCeiling = 2.0
	for i, p := range presets {
		if i == 0 {
			continue
		}
		if rows[i] > regressionCeiling*rows[0] {
			t.Errorf("%s dot error %.4f is more than %.1fx tq3 %.4f on symmetric-K — unsafe as drop-in for existing tq3 use cases", p.Name, rows[i], regressionCeiling, rows[0])
		}
	}
}
