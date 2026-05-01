package turboquant

import (
	"math"
	"math/rand/v2"
	"testing"
)

// TestBlockRoundTripWithZero verifies that a Block carrying a non-zero Zero
// offset round-trips through MarshalBinary / UnmarshalBinary intact. The Zero
// field was added to the v7 block format alongside the channel bitmap to
// support asymmetric-primary presets; serialising it correctly is a
// prerequisite for everything the asymmetric path does downstream.
func TestBlockRoundTripWithZero(t *testing.T) {
	block := Block{
		Version:        BlockVersion,
		PresetID:       PresetTQ3.ID,
		Role:           uint8(roleKey),
		Objective:      uint8(objectiveMSE),
		OriginalDim:    4,
		PaddedDim:      4,
		BlockDim:       4,
		RegularBits:    3,
		RotationSeed:   99,
		CodebookID:     3,
		QJLRows:        0,
		AuxLayoutID:    1,
		Scale:          1.25,
		Zero:           -0.375, // representative non-zero offset
		RegularIndices: []byte{0b10101010},
	}
	raw, err := block.MarshalBinary()
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var got Block
	if err := got.UnmarshalBinary(raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if got.Scale != block.Scale {
		t.Errorf("Scale: got %f want %f", got.Scale, block.Scale)
	}
	if got.Zero != block.Zero {
		t.Errorf("Zero: got %f want %f", got.Zero, block.Zero)
	}
}

// TestAsymmetricPrimaryOnBiasedVector checks that the centred-asymmetric
// encode+decode path recovers a vector whose coordinates share a large
// per-channel mean — the failure mode that motivates the asymmetric-primary
// scheme in the first place. The symmetric path is included as a control: it
// should produce a materially larger reconstruction error on the same input.
func TestAsymmetricPrimaryOnBiasedVector(t *testing.T) {
	const dim = 64
	rng := rand.New(rand.NewPCG(0xC0DE, 0xCAFE))

	// Biased vector: small Gaussian noise plus a large constant per-channel mean.
	// After rotation the mean is spread across all coordinates; without the
	// zero offset the symmetric codebook has to represent a non-zero-mean
	// distribution and burns resolution.
	v := make([]float32, dim)
	for i := range v {
		v[i] = float32(rng.NormFloat64())*0.1 + 2.5
	}

	reconstruct := func(p Preset) []float32 {
		ev, err := EncodeKeyVector(v, p)
		if err != nil {
			t.Fatalf("encode %s: %v", p.Name, err)
		}
		raw, err := ev.MarshalBinary()
		if err != nil {
			t.Fatalf("marshal %s: %v", p.Name, err)
		}
		out, _, err := DecodeVector(raw)
		if err != nil {
			t.Fatalf("decode %s: %v", p.Name, err)
		}
		return out
	}

	l2 := func(a, b []float32) float64 {
		var s float64
		for i := range a {
			d := float64(a[i] - b[i])
			s += d * d
		}
		return math.Sqrt(s)
	}

	symErr := l2(v, reconstruct(PresetTQ3))
	asymErr := l2(v, reconstruct(PresetTQ3A))
	t.Logf("biased-vector L2 recon err:  tq3 = %.4f   tq3a = %.4f   (lower better)", symErr, asymErr)

	if asymErr >= symErr {
		t.Errorf("tq3a (%.4f) did not improve on tq3 (%.4f) for biased vector", asymErr, symErr)
	}
	// Demand at least a 15% relative improvement — matches the floor used in
	// TestQJLRecoversAsymmetricKey. The absolute residual error after
	// centred-asymmetric is driven by codebook-step × scale, both of which
	// scale with the rotated coordinates' magnitude; asserting a relative
	// rather than absolute bound keeps the test robust to changes in
	// rotation-seed or codebook layout.
	improvement := (symErr - asymErr) / symErr
	if improvement < 0.15 {
		t.Errorf("tq3a improvement over tq3 was only %.1f%% — below the 15%% floor", 100*improvement)
	}
}
