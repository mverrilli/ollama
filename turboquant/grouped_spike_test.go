package turboquant

import (
	"math"
	"math/rand/v2"
	"testing"
)

// TestGroupedScalesNoBenefit is the documented negative result for
// per-group (scale, zero) quantization — the medenijazbec fork's
// "PerGroupScaleCount" / "NativeGroupedHeader" idea (see
// turboquant/native_layout.go in turboquant-0.18.3) where each per-128 (or
// finer) sub-region of the rotated vector carries its own (scale, zero)
// pair instead of one per block.
//
// The test compares the existing tq3qa pipeline (one (scale, zero) per
// block, plus outlier split + QJL residual) against a simulated grouped-
// asymmetric path at groupSize ∈ {128, 64, 32, 16, 8} on the same
// Qwen-2-style synthetic distribution the qjl_recon tests use. All grouped
// configurations come out *worse* than tq3qa, even at the most aggressive
// groupSize=8 (which would cost 120 extra bytes per block).
//
// Why: random-orthogonal Householder QR rotation evenly distributes
// variance across the rotated coordinates by design, so per-group
// statistics capture no additional structure. The error budget is
// dominated by outlier channels (handled by the outlier split) and
// quantization residual (handled by QJL), not by per-coordinate-group
// variance differences. Adding grouped scales costs memory and gains
// nothing on this distribution.
//
// If a future change can show grouped scales improving on tq3qa for some
// concrete model's K distribution, this test's expectation will need to
// be updated. Until then it serves as evidence the idea was considered
// and rejected on data, not assumption.
func TestGroupedScalesNoBenefit(t *testing.T) {
	const (
		headDim = 128
		trials  = 128
		bits    = 3
	)

	// Fixed query across all configurations — the comparison is preset-vs-preset
	// not query-dependent.
	q := func() []float32 {
		rng := rand.New(rand.NewPCG(0xfeedface, 0xc0debabe))
		v := make([]float32, headDim)
		for i := range v {
			v[i] = float32(rng.NormFloat64())
		}
		return v
	}()

	codebook, boundaries := scalarCodebook(headDim, bits)
	rotation := BuildRotation(headDim, PresetTQ3.RotationSeed)

	// groupedRecon simulates a hypothetical grouped-asymmetric encode/decode:
	// split rotated vector into chunks of `groupSize`, encode each with its
	// own (mean, RMS-of-centred), reconstruct. No outlier split, no QJL.
	// Isolates "what does sub-block grouping alone buy?".
	groupedRecon := func(v []float32, groupSize int) []float32 {
		rotated := ApplyRotation(v, rotation)
		recon := make([]float32, headDim)
		for off := 0; off < headDim; off += groupSize {
			end := off + groupSize
			if end > headDim {
				end = headDim
			}
			group := rotated[off:end]
			zero, scale := asymmetricBlockStats(group)
			for i, val := range group {
				if scale == 0 {
					recon[off+i] = zero
					continue
				}
				idx := quantizeScalarByBoundary((val-zero)/scale, codebook, boundaries)
				recon[off+i] = dequantizeScalar(idx, codebook)*scale + zero
			}
		}
		return ApplyInverseRotation(recon, rotation)
	}

	// Helper: re-seed and re-emit the same trial vectors against a recon fn.
	dotErrFor := func(reconFn func(v []float32) []float32) float64 {
		r := rand.New(rand.NewPCG(0x9e3779b97f4a7c15, 0xbf58476d1ce4e5b9))
		mk := func() []float32 {
			bias := make([]float32, headDim)
			for i := range bias {
				bias[i] = 0.25 * float32(r.NormFloat64())
			}
			for k := 0; k < 8; k++ {
				idx := r.IntN(headDim)
				bias[idx] += float32(math.Copysign(2.0+r.Float64(), r.NormFloat64()))
			}
			v := make([]float32, headDim)
			for i := range v {
				v[i] = float32(r.NormFloat64()) + bias[i]
			}
			return v
		}
		var total float64
		for k := 0; k < trials; k++ {
			v := mk()
			rec := reconFn(v)
			var gold, est float64
			for i := range v {
				gold += float64(q[i]) * float64(v[i])
				est += float64(q[i]) * float64(rec[i])
			}
			total += math.Abs(est - gold)
		}
		return total / float64(trials)
	}

	// Baseline: tq3qa via the actual encode/decode pipeline.
	tq3qaErr := dotErrFor(func(v []float32) []float32 {
		ev, err := EncodeKeyVector(v, PresetTQ3QA)
		if err != nil {
			t.Fatal(err)
		}
		raw, err := ev.MarshalBinary()
		if err != nil {
			t.Fatal(err)
		}
		rec, _, err := DecodeVector(raw)
		if err != nil {
			t.Fatal(err)
		}
		return rec
	})

	t.Logf("baseline tq3qa (outlier+QJL+asymmetric): dot-err = %.4f", tq3qaErr)
	t.Logf("  %-9s %-10s %-18s %s", "groupSize", "dot-err", "extra-bytes/block", "vs tq3qa")

	groupSizes := []int{128, 64, 32, 16, 8}
	beatsBaseline := false
	for _, gs := range groupSizes {
		err := dotErrFor(func(v []float32) []float32 { return groupedRecon(v, gs) })
		extra := (headDim/gs - 1) * 8 // 8 bytes per (scale, zero) f32 pair
		rel := (err - tq3qaErr) / tq3qaErr * 100
		t.Logf("  %-9d %-10.4f %-18d %+.1f%%", gs, err, extra, rel)
		if err < tq3qaErr {
			beatsBaseline = true
		}
	}

	if beatsBaseline {
		t.Errorf("a grouped-asymmetric configuration beat tq3qa — re-evaluate whether per-group scales are worth adding")
	}
}
