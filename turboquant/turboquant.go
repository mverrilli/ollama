package turboquant

import (
	"fmt"
)

// BlockVersion 7 introduces two layout changes to the v6 Block format:
//
//  1. ChannelIndices []uint16 → ChannelBitmap []byte. One bit per channel
//     of the full (pre-split) vector, set iff that channel belongs to this
//     sub-block. At headDim=128 with a 32-channel outlier split, 16 bytes
//     of bitmap per sub-block replaces 192 + 64 = 256 bytes of indices —
//     ~240 bytes saved per encoded vector pair, only on the CPU /
//     offline-serialised path used by unit tests and dump-and-replay
//     tooling. The GPU kernel layout amortises outlier indices across
//     cells per-head independently of this format and is unaffected.
//
//  2. New Zero float32 field per Block, supporting centred-asymmetric
//     primary quantization (Preset.AsymmetricPrimary). Symmetric presets
//     write 0; decoding is unconditional, so legacy symmetric blocks
//     decode bit-identically to v6.
//
// Both changes ship under the same v7 bump because the bitmap rework was
// itself unreleased — there is no intermediate v7-without-Zero in the
// wild that would need back-compat handling.
const BlockVersion = 7

type vectorRole uint8

const (
	roleGeneric vectorRole = iota
	roleKey
	roleValue
)

type vectorObjective uint8

const (
	objectiveMSE vectorObjective = iota + 1
	objectiveProduct
)

// QuantScheme identifies the high-level quantization algorithm for a preset.
// The zero value (SchemeHouseholderLloydMax) is the TurboQuant rotation +
// codebook path. Non-zero values select lighter-weight integer paths that
// share the TurboQuantCache infrastructure but skip rotation and codebooks.
type QuantScheme uint8

const (
	// SchemeHouseholderLloydMax: random Householder QR rotation followed by
	// per-head Lloyd-Max scalar quantization. All tq* presets use this scheme.
	SchemeHouseholderLloydMax QuantScheme = iota

	// SchemeQ8K: per-group asymmetric int8, no rotation, group size 32.
	// ~9 bits/element effective. Works on all models including Qwen2.5.
	SchemeQ8K

	// SchemeQ4K: per-group asymmetric int4 (nibble), no rotation, group size 32.
	// ~5 bits/element effective. Not suitable for models with large K bias
	// (e.g. Qwen2.5); safe for llama3.x and qwen3.
	SchemeQ4K
)

type Preset struct {
	ID             uint8
	Name           string
	RotationSeed   uint64
	KeyPrimaryBits int
	ValueBits      int
	QJLRowsDivisor int
	OutlierBits    int
	OutlierCount   int
	// AsymmetricPrimary, when true, centres each per-block rotated vector by
	// its mean before scalar quantization and stores that mean in Block.Zero.
	// Decoding unconditionally adds Zero back. Targets models whose learned
	// K bias produces a non-zero-mean rotated distribution (Qwen 2 family).
	AsymmetricPrimary bool

	// Scheme identifies which quantization algorithm this preset uses.
	// Zero value (SchemeHouseholderLloydMax) is the TurboQuant path.
	// SchemeQ8K and SchemeQ4K select the per-group integer paths.
	Scheme QuantScheme
}

var (
	// All four tq* presets ship with OutlierCount=0 (pure uniform Lloyd-Max
	// after Householder QR rotation, i.e. the core of TurboQuant Algorithm 1
	// §3.1 without the optional outlier split from §4.3 or the QJL residual
	// sketch from Algorithm 2). The uniform defaults were chosen after
	// measuring that on the models this fork ships against (llama, gemma3,
	// qwen3-coder), outlier split hurts both decode throughput and PPL — the
	// paper's split targets heavy-tailed rotated K distributions, which these
	// models don't exhibit, and the extra metadata (92 vs 52 bytes/head/cell
	// at oc=32) translates to ~25% decode regression on 3B-class models at
	// short context. Keeping the defaults symmetric across tq2 / tq3 / tq2k /
	// tq3k means the digit in the preset name maps directly to effective
	// bits/elem: "tq3" is exactly 3 bits, not the 3.25 bits you'd get under
	// outlier split with oc=32.
	//
	// The outlier-split kernel path remains in the code (encode/dequant
	// dispatchers check op_params[2..3] and route to the outlier variant when
	// both are non-zero). A future dynamic-dispatch PR can enable it per
	// model / per env var — e.g. for models with larger headDim where the
	// metadata overhead amortizes better. See project_tq_backlog.md for
	// the planned dynamic-oc dispatch work.

	// tq2: 2-bit K + 2-bit V, both rotated and Lloyd-Max quantized. Highest
	// compression tier — effective 2 bits/elem both sides.
	PresetTQ2 = newPreset(1, "tq2", 2, 2, 1, 0x25c0ffee, 3, 0)

	// tq3: 3-bit K + 3-bit V, both rotated and Lloyd-Max quantized. Default
	// "balanced" tier — effective 3 bits/elem both sides.
	PresetTQ3 = newPreset(2, "tq3", 3, 3, 1, 0x35c0ffee, 4, 0)

	// tq3k: 3-bit K only, V stays as f16. ~40% KV VRAM savings with near-f16
	// decode (no V dequant at all). ValueBits=0 signals K-only mode to the
	// kvcache layer.
	PresetTQ3K = newPreset(3, "tq3k", 3, 0, 1, 0x35c0ffee, 4, 0)

	// tq2k: 2-bit K only, V stays as f16. Maximum K compression with f16 V;
	// smallest K footprint before PPL degrades too much. ValueBits=0 signals
	// K-only mode to the kvcache layer.
	PresetTQ2K = newPreset(4, "tq2k", 2, 0, 1, 0x25c0ffee, 3, 0)

	// tq3q enables the outlier-split + 1-bit QJL residual sketch path on
	// the CPU block-protocol encoder. Not wired into the GPU kv-cache pipeline.
	PresetTQ3Q = newPreset(5, "tq3q", 3, 3, 1, 0x35c0ffee, 4, 32)

	// tq3a and tq3ka enable per-vector centred-asymmetric primary
	// quantization (see Preset.AsymmetricPrimary). They address the Qwen 2
	// weak spot by centring each rotated vector on its mean before scalar
	// quantization — the symmetric Lloyd-Max codebook doesn't waste
	// resolution on the empty half of a biased distribution. Fully wired on
	// CUDA: encode, dequant, and fused fattn all handle the asymmetric zeros
	// path. Metal falls back to f16 (asymmetric primary is CUDA-only).
	PresetTQ3A  = newAsymmetricPreset(7, "tq3a", 3, 3, 1, 0x35c0ffee, 4, 0)
	PresetTQ3KA = newAsymmetricPreset(8, "tq3ka", 3, 0, 1, 0x35c0ffee, 4, 0)

	// tq3qa: asymmetric primary quantization + outlier-split + QJL at 3-bit K+V.
	// The qjl_recon_test.go sweep {tq3, tq3a, tq3q, tq3qa} measures whether QJL
	// still helps once asymmetric primary has absorbed the mean-shift.
	PresetTQ3QA = newAsymmetricPreset(9, "tq3qa", 3, 3, 1, 0x35c0ffee, 4, 32)

	// 2-bit variants. tq2q, tq2a / tq2ka, tq2qa mirror the 3-bit naming.
	// The 4-variant sweep test runs at both bit widths.
	PresetTQ2A  = newAsymmetricPreset(11, "tq2a", 2, 2, 1, 0x25c0ffee, 3, 0)
	PresetTQ2KA = newAsymmetricPreset(12, "tq2ka", 2, 0, 1, 0x25c0ffee, 3, 0)
	PresetTQ2Q  = newPreset(13, "tq2q", 2, 2, 1, 0x25c0ffee, 3, 32)
	PresetTQ2QA = newAsymmetricPreset(15, "tq2qa", 2, 2, 1, 0x25c0ffee, 3, 32)

	// tq4: 4-bit Lloyd-Max K + 4-bit V, both rotation + codebook quantized.
	// 16-entry symmetric codebook. Higher fidelity than tq3 at the cost of
	// ~33% more KV storage. Well within the range validated by the TurboQuant
	// paper (Gaussian mixture calibration at 4 bits).
	PresetTQ4 = newPreset(25, "tq4", 4, 4, 1, 0x45c0ffee, 5, 0)

	// tq4k: 4-bit K only, V at f16. Lower storage overhead than tq4 while
	// keeping the full 16-entry K codebook fidelity.
	PresetTQ4K = newPreset(26, "tq4k", 4, 0, 1, 0x45c0ffee, 5, 0)

	// tq4a and tq4ka: asymmetric (mean-centred) primary quantization at 4 bits.
	// Intended for models with K projection bias (Qwen2/3 family).
	PresetTQ4A  = newAsymmetricPreset(27, "tq4a", 4, 4, 1, 0x45c0ffee, 5, 0)
	PresetTQ4KA = newAsymmetricPreset(28, "tq4ka", 4, 0, 1, 0x45c0ffee, 5, 0)

	// tq4qa: full stack — asymmetric + outlier split + QJL at 4-bit K+V.
	PresetTQ4QA = newAsymmetricPreset(29, "tq4qa", 4, 4, 1, 0x45c0ffee, 5, 32)

	// q8k: per-group asymmetric int8 (group=32), K-only, no rotation.
	// ~9 bits/element. Works on all models including Qwen2.5 (large K bias
	// is safely representable with 256 levels and per-group scales).
	PresetQ8K = newQ8KPreset(30, "q8k", 8, 0)

	// q8kv: per-group asymmetric int8 (group=32), K+V, no rotation.
	PresetQ8KV = newQ8KPreset(31, "q8kv", 8, 8)

	// q4k: per-group asymmetric int4 (nibble, group=32), K-only, no rotation.
	// ~5 bits/element. Not suitable for models with large K projection bias
	// (Qwen2.5): use q8k instead for those.
	PresetQ4K = newQ4KPreset(32, "q4k", 4, 0)

	// q4kv: per-group asymmetric int4 (nibble, group=32), K+V, no rotation.
	PresetQ4KV = newQ4KPreset(33, "q4kv", 4, 4)
)

func newPreset(id uint8, name string, keyBits int, valueBits int, qjlRowsDivisor int, seed uint64, outlierBits int, outlierCount int) Preset {
	return Preset{
		ID:             id,
		Name:           name,
		RotationSeed:   seed,
		KeyPrimaryBits: keyBits,
		ValueBits:      valueBits,
		QJLRowsDivisor: qjlRowsDivisor,
		OutlierBits:    outlierBits,
		OutlierCount:   outlierCount,
	}
}

// newAsymmetricPreset constructs a preset with the same knobs as newPreset
// plus AsymmetricPrimary=true. All six asymmetric-variant presets use this
// constructor; the asymmetric flag is orthogonal to OutlierCount / QJLRows.
func newAsymmetricPreset(id uint8, name string, keyBits int, valueBits int, qjlRowsDivisor int, seed uint64, outlierBits int, outlierCount int) Preset {
	p := newPreset(id, name, keyBits, valueBits, qjlRowsDivisor, seed, outlierBits, outlierCount)
	p.AsymmetricPrimary = true
	return p
}

// newQ8KPreset constructs a per-group int8 preset (SchemeQ8K). The rotation
// seed and Lloyd-Max fields are zeroed — they are irrelevant for this scheme.
func newQ8KPreset(id uint8, name string, keyBits, valueBits int) Preset {
	return Preset{
		ID:             id,
		Name:           name,
		KeyPrimaryBits: keyBits,
		ValueBits:      valueBits,
		Scheme:         SchemeQ8K,
	}
}

// newQ4KPreset constructs a per-group int4 (nibble) preset (SchemeQ4K).
func newQ4KPreset(id uint8, name string, keyBits, valueBits int) Preset {
	return Preset{
		ID:             id,
		Name:           name,
		KeyPrimaryBits: keyBits,
		ValueBits:      valueBits,
		Scheme:         SchemeQ4K,
	}
}

func (p Preset) HasOutlierSplit() bool {
	return p.OutlierBits > 0 && p.OutlierCount > 0
}

// HasAsymmetricPrimary reports whether this preset uses centred-asymmetric
// primary quantization (mean offset per block, stored in Block.Zero) rather
// than the default symmetric path.
func (p Preset) HasAsymmetricPrimary() bool {
	return p.AsymmetricPrimary
}

// PresetByName resolves a user-facing preset string (the values an end user
// can pass via OLLAMA_KV_CACHE_TYPE) to its Preset definition. The exposed
// set:
//
//   - tq2, tq3, tq2k, tq3k     — shipped baseline tiers
//   - tq2qa, tq3qa, tq4qa      — full stack (asymmetric + outliers + QJL), K+V
//
// The single-mechanism intermediates (tq3a, tq3q, tq2a, tq2q, plus K-only
// variants) are not exposed here; the unit tests show the stacked *qa
// combinations dominate them on the target distribution. They remain
// reachable as exported package vars (PresetTQ3A, PresetTQ2Q, etc.) for
// ablation testing only — see qjl_recon_test.go.
func PresetByName(name string) (Preset, error) {
	switch name {
	case "tq2":
		return PresetTQ2, nil
	case "tq3":
		return PresetTQ3, nil
	case "tq3k":
		return PresetTQ3K, nil
	case "tq2k":
		return PresetTQ2K, nil
	case "tq3qa":
		return PresetTQ3QA, nil
	case "tq2qa":
		return PresetTQ2QA, nil
	case "tq4":
		return PresetTQ4, nil
	case "tq4k":
		return PresetTQ4K, nil
	case "tq4a":
		return PresetTQ4A, nil
	case "tq4ka":
		return PresetTQ4KA, nil
	case "tq4qa":
		return PresetTQ4QA, nil
	case "q8k":
		return PresetQ8K, nil
	case "q8kv":
		return PresetQ8KV, nil
	case "q4k":
		return PresetQ4K, nil
	case "q4kv":
		return PresetQ4KV, nil
	default:
		return Preset{}, fmt.Errorf("unknown turboquant preset %q", name)
	}
}

func PresetByID(id uint8) (Preset, error) {
	switch id {
	case PresetTQ2.ID:
		return PresetTQ2, nil
	case PresetTQ3.ID:
		return PresetTQ3, nil
	case PresetTQ3K.ID:
		return PresetTQ3K, nil
	case PresetTQ2K.ID:
		return PresetTQ2K, nil
	case PresetTQ3Q.ID:
		return PresetTQ3Q, nil
	case PresetTQ3A.ID:
		return PresetTQ3A, nil
	case PresetTQ3KA.ID:
		return PresetTQ3KA, nil
	case PresetTQ3QA.ID:
		return PresetTQ3QA, nil
	case PresetTQ2A.ID:
		return PresetTQ2A, nil
	case PresetTQ2KA.ID:
		return PresetTQ2KA, nil
	case PresetTQ2Q.ID:
		return PresetTQ2Q, nil
	case PresetTQ2QA.ID:
		return PresetTQ2QA, nil
	case PresetTQ4.ID:
		return PresetTQ4, nil
	case PresetTQ4K.ID:
		return PresetTQ4K, nil
	case PresetTQ4A.ID:
		return PresetTQ4A, nil
	case PresetTQ4KA.ID:
		return PresetTQ4KA, nil
	case PresetTQ4QA.ID:
		return PresetTQ4QA, nil
	default:
		return Preset{}, fmt.Errorf("unknown turboquant preset id %d", id)
	}
}

func (p Preset) KeyQJLRows(dim int) int {
	if dim <= 0 {
		return 0
	}
	if p.QJLRowsDivisor <= 0 {
		return 0
	}
	rows := dim / p.QJLRowsDivisor
	if rows < 1 {
		return 1
	}
	return rows
}
