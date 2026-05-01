package kvcache

import (
	"fmt"
	"log/slog"
	"math"
	"sync"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model/input"
	"github.com/ollama/ollama/turboquant"
)

type TurboQuantCache struct {
	meta      *Causal
	preset    turboquant.Preset
	isReserve bool

	compressedK ml.TQCompressedKManager

	// phase2Checked ensures GPU encode is activated at most once.
	phase2Checked bool

	headDim    int
	numKVHeads int

	// encodeResults stores per-layer EncodeK result tensors for the current
	// forward pass. DequantK uses them as src[0] to establish the graph
	// dependency (encode before dequant in the ggml scheduler).
	encodeResults map[int]ml.Tensor

	// vEncodeResults stores per-layer EncodeV result tensors for the current
	// forward pass. DequantV uses them to establish the encode→dequant ordering.
	vEncodeResults map[int]ml.Tensor

	// logPathOnce ensures each active Get() path is logged at most once per
	// cache instance (avoids log spam: Get() is called every layer every step).
	logPathOnce [5]sync.Once

	// fusedFallbackEligible gates the inline-decode fused-FA fallback paths
	// (Get paths 2 and 4). The CUDA fused kernel is template-instantiated only
	// at D=128, so models with a larger head dim (gemma4 D=512) must skip it
	// to avoid a kernel-side GGML_ASSERT. The Metal fused kernel has both
	// D=128 and D=256 variants (kernel_tq_fattn_vec_*{,_d256}), so gemma3
	// D=256 is eligible on Metal but not on CUDA until the CUDA kernel gains
	// a D=256 instantiation. The DequantK + stock FA path (Get paths 0/1/5)
	// works at any head dim — this gate is specific to the inline-decode
	// variants.
	fusedFallbackEligible bool

	// preferFusedAttn is true on Metal. At long context, DequantKV + stock FA
	// writes a full f16 intermediate buffer (doubling KV bandwidth) before
	// attention reads it. The fused kernel (kernel_tq_fattn_vec_packed) reads
	// packed K+V directly and skips the intermediate write — dramatically
	// faster on Metal for *decode* (Q=1). On CUDA the DequantKV path is
	// preferred because the intermediate buffer stays in L2 and stock flash
	// attention is highly tuned.
	//
	// For prefill (Q>1, see curQueryLen) DequantKV wins on every backend
	// because it decodes each packed cell once then runs stock FA with full
	// batch amortisation, whereas the fused kernel re-decodes each cell per
	// Q-token. Get() uses the combination of preferFusedAttn and curQueryLen
	// to route prefill vs decode independently.
	preferFusedAttn bool

	// curQueryLen is the number of query tokens in the current forward pass,
	// captured at StartForward from len(batch.Positions). curQueryLen==1
	// means decode; curQueryLen>1 means prefill (or a batched prompt chunk).
	curQueryLen int

	// rotMatrix is the R^T rotation matrix sized for this cache's headDim.
	// Set in activateGPUEncode. Get() sets the backend's tqRotationMatrix to
	// this value per-call (consume-once) right before returning rotated K.
	rotMatrix ml.Tensor

	// vRotMatrix is the R (inverse) matrix used to undo V rotation after
	// attention in K+V presets (tq3/tq2). Nil for K-only presets. Get() sets
	// the backend's tqVRotationMatrix to this value per-call along with
	// rotMatrix so SDPA applies R @ attn_out after flash attention.
	vRotMatrix ml.Tensor

	// rotSetter is the cached type assertion of c.meta.backend onto the TQ
	// rotation setter interface, populated once in activateGPUEncode. nil
	// when the backend doesn't support TQ rotation hooks (the fallback case).
	rotSetter tqRotSetter

	// pendingKBiases buffers K projection biases set by SetLayerKBias before
	// activateGPUEncode has initialised compressedK. Applied to compressedK
	// once the manager is created.
	pendingKBiases map[int]ml.Tensor

	// reserveSkipVPlaceholder, when true, suppresses the SkipV synthesis block
	// in the reserve graph for the current Get() call. Set when the K reserve
	// branch placed a K+V fused tensor (which carries V internally as vPacked,
	// so the inference-time value return is nil). The reserve graph must
	// mirror that — synthesising a Zeros f16 V here would re-introduce the
	// scratch buffer the K+V fused path was meant to eliminate. Read-and-cleared
	// once per Get() call.
	reserveSkipVPlaceholder bool
}

// tqRotSetter is the backend hook TurboQuantCache uses to arm the per-call
// rotation matrices SDPA consumes. Implemented by ml/backend/ggml.Backend.
type tqRotSetter interface {
	SetTQRotationMatrix(ml.Tensor)
	SetTQVRotationMatrix(ml.Tensor)
}

// isSWACausal reports whether a *Causal has sliding-window attention
// active. Plain Causal caches have swaWindowSize either 0 (before Init
// normalizes the default) or math.MaxInt32 (after); SWA constructors set
// it to the actual window size.
func isSWACausal(c *Causal) bool {
	return c.swaWindowSize > 0 && c.swaWindowSize != math.MaxInt32
}

// AttentionKVWrapper is implemented by caches that embed *Recurrent and
// expose the attention half of a hybrid (SSM/recurrent + attention) cache.
// WrapWithTurboQuant uses it to inject TurboQuant compression into the
// attention KV path without disturbing conv/recurrent state buffers.
// *kvcache.Recurrent implements this interface, so any model HybridCache that
// embeds *Recurrent satisfies it automatically via Go method promotion.
type AttentionKVWrapper interface {
	AttentionKV() *Causal
	SetAttentionKV(Cache)
}

// WrapWithTurboQuant returns a cache that applies TurboQuant compression to
// global-attention Causal layers and a bool reporting whether any wrapping
// took effect. For a top-level *Causal (non-SWA), it returns a new
// *TurboQuantCache. For a *WrapperCache, it mutates the caches slice in
// place, replacing every non-SWA *Causal sub-cache with a *TurboQuantCache,
// and returns the same *WrapperCache pointer. This enables TQ on SWA models
// like gemma3/gemma4 where the global attention layers dominate KV memory
// at long context. Returns (cache, false) if no eligible sub-caches were
// found.
func WrapWithTurboQuant(cache Cache, preset turboquant.Preset) (Cache, bool) {
	switch c := cache.(type) {
	case *Causal:
		// Reject SWA caches. Plain NewCausalCache leaves swaWindowSize=0
		// until Init() normalizes it to math.MaxInt32; SWA constructors set
		// it to the actual window size. "Plain causal" means the field is
		// either 0 (uninitialized default) or math.MaxInt32 (post-Init).
		if isSWACausal(c) {
			slog.Warn("turboquant: top-level Causal is sliding-window, cannot wrap")
			return cache, false
		}
		return &TurboQuantCache{
			meta:           c,
			preset:         preset,
			encodeResults:  make(map[int]ml.Tensor),
			vEncodeResults: make(map[int]ml.Tensor),
		}, true

	case *WrapperCache:
		// Mutate sub-caches in place: replace every non-SWA *Causal with a
		// *TurboQuantCache wrapping it. SWA sub-caches (SWACache, SWAMemCache)
		// are left untouched — they still allocate f16 K/V as before.
		wrapped := 0
		for i, sub := range c.caches {
			inner, ok := sub.(*Causal)
			if !ok || isSWACausal(inner) {
				continue
			}
			c.caches[i] = &TurboQuantCache{
				meta:           inner,
				preset:         preset,
				encodeResults:  make(map[int]ml.Tensor),
				vEncodeResults: make(map[int]ml.Tensor),
			}
			wrapped++
		}
		if wrapped == 0 {
			slog.Warn("turboquant: no eligible Causal sub-caches in WrapperCache, falling back to unwrapped cache")
			return cache, false
		}
		slog.Info("turboquant: wrapped Causal sub-caches inside WrapperCache",
			"count", wrapped, "preset", preset.Name)
		return cache, true

	case AttentionKVWrapper:
		inner := c.AttentionKV()
		if inner == nil {
			slog.Warn("turboquant: hybrid cache inner kv is not *Causal (already wrapped?), leaving as-is")
			return cache, false
		}
		if isSWACausal(inner) {
			slog.Warn("turboquant: hybrid cache inner *Causal is sliding-window, cannot wrap")
			return cache, false
		}
		c.SetAttentionKV(&TurboQuantCache{
			meta:           inner,
			preset:         preset,
			encodeResults:  make(map[int]ml.Tensor),
			vEncodeResults: make(map[int]ml.Tensor),
		})
		slog.Info("turboquant: wrapped attention KV in hybrid recurrent cache",
			"preset", preset.Name)
		return cache, true

	default:
		slog.Warn("turboquant: underlying cache is not *Causal or *WrapperCache, falling back to unwrapped cache")
		return cache, false
	}
}

func (c *TurboQuantCache) Init(backend ml.Backend, dtype ml.DType, maxSequences, capacity, maxBatch int) {
	// K is always compressed; suppress inner Causal from allocating it.
	c.meta.SkipK = true
	// V is compressed when ValueBits > 0; skip the inner Causal V allocation.
	if c.preset.ValueBits > 0 {
		c.meta.SkipV = true
	}
	c.meta.Init(backend, ml.DTypeF16, maxSequences, capacity, maxBatch)
	slog.Info("turboquant cache initialized", "preset", c.preset.Name,
		"K_bits", c.preset.KeyPrimaryBits, "V_bits", c.preset.ValueBits)
}

func (c *TurboQuantCache) Close() {
	if c.compressedK != nil {
		c.compressedK.Close()
		c.compressedK = nil
	}
	c.meta.Close()
}

func (c *TurboQuantCache) SetLayer(layer int)              { c.meta.SetLayer(layer) }
func (c *TurboQuantCache) SetConfig(config ml.CacheConfig) { c.meta.SetConfig(config) }

// SetLayerKBias passes the K projection bias tensor for the given layer to the
// TQ compression manager. Called once per layer at model init for architectures
// that include a bias on the K projection (e.g. Qwen2). The bias is subtracted
// from K before rotation in the encoder; attention correctness is preserved
// because a constant shift in K cancels out in softmax.
func (c *TurboQuantCache) SetLayerKBias(layer int, bias ml.Tensor) {
	if bias == nil {
		return
	}
	if c.compressedK == nil {
		// activateGPUEncode hasn't run yet — buffer for later application.
		if c.pendingKBiases == nil {
			c.pendingKBiases = make(map[int]ml.Tensor)
		}
		c.pendingKBiases[layer] = bias
		return
	}
	type kBiasSetter interface {
		SetLayerKBias(layer int, bias ml.Tensor)
	}
	if kbs, ok := c.compressedK.(kBiasSetter); ok {
		kbs.SetLayerKBias(layer, bias)
	}
}

func (c *TurboQuantCache) StartForward(ctx ml.Context, batch input.Batch, reserve bool) error {
	c.isReserve = reserve
	c.curQueryLen = len(batch.Positions)
	clear(c.encodeResults)
	clear(c.vEncodeResults)
	return c.meta.StartForward(ctx, batch, reserve)
}

func (c *TurboQuantCache) Put(ctx ml.Context, key, value ml.Tensor) {
	// Capture headDim early — needed even during reserve to size placeholders.
	if c.headDim == 0 && key != nil {
		c.headDim = key.Dim(0)
		c.numKVHeads = key.Dim(1)
	}

	// Activate the GPU encode path on the first Put (reserve or not) once we
	// know headDim/numKVHeads. Doing this during reserve lets EnsureLayer run
	// on the reserve pass, which books TQ's per-layer persistent K/V buffers
	// into btDeviceMemory.Cache[layer] (via newTensor's ctx.layer accounting
	// in EnsureLayer). Without this, the scheduler's fit/alloc probe sees a
	// 0-byte Cache for tq2/tq3 and only the f16 V half for tq2k/tq3k, which
	// is wrong for both headline-footprint reporting and long-context fit math.
	if !c.phase2Checked && c.headDim > 0 {
		c.phase2Checked = true
		c.activateGPUEncode()
	}

	if c.isReserve {
		c.meta.Put(ctx, key, value)
		// Eagerly allocate TQ/i4 persistent buffers during reserve so the
		// scheduler's per-layer Cache totals reflect the real post-compression
		// footprint. Also create the encode graph node so the reserve graph has
		// the same K-branch structure as the inference graph. Without the encode
		// node, gallocr's live-range analysis sees a truncated graph and
		// over-allocates scratch (measured: +52 MiB for i4k, +222 MiB for tq*qa).
		if c.compressedK != nil {
			layer := c.meta.curLayer
			capacity := len(c.meta.cells)
			c.compressedK.EnsureLayer(layer, capacity)
			if c.preset.ValueBits > 0 {
				c.compressedK.EnsureVLayer(layer, capacity)
				kResult, vResult := c.compressedK.EncodeKV(ctx, layer, key, value, 0)
				if kResult != nil {
					ctx.Forward(kResult)
					c.encodeResults[layer] = kResult
					c.vEncodeResults[layer] = vResult
				}
			} else {
				encodeResult := c.compressedK.EncodeK(ctx, layer, key, 0)
				if encodeResult != nil {
					ctx.Forward(encodeResult)
					c.encodeResults[layer] = encodeResult
				}
			}
		}
		return
	}

	if c.compressedK != nil {
		layer := c.meta.curLayer
		capacity := len(c.meta.cells)

		c.compressedK.EnsureLayer(layer, capacity)
		if c.preset.ValueBits > 0 {
			c.compressedK.EnsureVLayer(layer, capacity)
		}

		// The TQ encode kernels (CUDA + Metal) write a contiguous run of
		// cells starting at firstCell. Causal.findLocs guarantees a
		// contiguous run when SkipK/SkipV is set (otherwise it returns
		// ErrKvCacheFull before we reach here); this loop is a defensive
		// invariant check that should never fire in practice.
		firstCell := 0
		if len(c.meta.curLocs) > 0 {
			firstCell = c.meta.curLocs[0]
			for i := 1; i < len(c.meta.curLocs); i++ {
				if c.meta.curLocs[i] != firstCell+i {
					panic(fmt.Sprintf("turboquant: non-contiguous cache slots %v — findLocs invariant violated", c.meta.curLocs))
				}
			}
		}

		if c.preset.ValueBits > 0 {
			// Combined K+V encode: single GGML op, two back-to-back kernels.
			kResult, vResult := c.compressedK.EncodeKV(ctx, layer, key, value, firstCell)
			if kResult != nil {
				ctx.Forward(kResult)
				c.encodeResults[layer] = kResult
				c.vEncodeResults[layer] = vResult
			}
		} else {
			// K-only presets (tq2k/tq3k): V stays as f16 in the Causal cache.
			encodeResult := c.compressedK.EncodeK(ctx, layer, key, firstCell)
			if encodeResult != nil {
				ctx.Forward(encodeResult)
				c.encodeResults[layer] = encodeResult
			}
		}

		// Inner Causal.Put() tracks cell metadata (positions, masks).
		// SkipK and SkipV suppress the actual K/V tensor writes.
		c.meta.Put(ctx, key, value)
		return
	}

	c.meta.Put(ctx, key, value)
}

func (c *TurboQuantCache) Get(ctx ml.Context) (ml.Tensor, ml.Tensor, ml.Tensor) {
	if c.isReserve {
		key, value, mask := c.meta.Get(ctx)
		// SkipK: synthesize K for graph sizing, matching the node type that the
		// inference path produces so gallocr pre-reserves the right scratch size.
		// Each branch mirrors exactly the inference routing in Get() below.
		if key == nil && c.headDim > 0 {
			nCells := 1
			if c.meta.curMask != nil {
				nCells = c.meta.curMask.Dim(0)
			}
			if c.compressedK != nil {
				layer := c.meta.curLayer
				enc := c.encodeResults[layer]
				switch {
				case c.preset.ValueBits > 0:
					// All tq* K+V presets: prefer fused K+V (no f16 scratch).
					// GetAsTQTensorKV succeeds for non-outlier and K+V-outlier presets;
					// returns false for K-only-outlier (blocked in fusedKernelSupports).
					// Falls back to DequantKV only when fused is unavailable (D!=128).
					vEnc := c.vEncodeResults[layer]
					if enc != nil && vEnc != nil {
						if gpuKV, ok := c.compressedK.GetAsTQTensorKV(ctx, layer, enc, vEnc, 0, nCells); ok {
							key = gpuKV
							// V carried inline — suppress f16 Zeros fallback in SkipV block.
							c.reserveSkipVPlaceholder = true
						}
						if key == nil {
							k, v := c.compressedK.DequantKV(ctx, layer, enc, vEnc, 0, nCells)
							if k != nil {
								key = k
							}
							if v != nil {
								value = v
							}
						}
						if key == nil {
							key = c.compressedK.DequantK(ctx, layer, enc, 0, nCells)
						}
						if value == nil {
							value = c.compressedK.DequantV(ctx, layer, vEnc, 0, nCells)
						}
					}
				case !c.preset.AsymmetricPrimary:
					// tq2k/tq3k/tq4k: prefer fused K-only (no scratch); fall back to DequantK.
					if enc != nil {
						if gpuKey, ok := c.compressedK.GetAsTQTensor(ctx, layer, enc, 0, nCells); ok {
							key = gpuKey
						} else {
							key = c.compressedK.DequantK(ctx, layer, enc, 0, nCells)
						}
					}
				default:
					// tq*ka: fused TQ kernel. Falls to DequantK if fused unavailable.
					if gpuKey, ok := c.compressedK.GetAsTQTensor(ctx, layer, enc, 0, nCells); ok {
						key = gpuKey
					} else if enc != nil {
						key = c.compressedK.DequantK(ctx, layer, enc, 0, nCells)
					}
				}
			}
			if key == nil {
				key = ctx.Input().Zeros(ml.DTypeF16, c.headDim, c.numKVHeads, nCells)
			}
		}
		// SkipV: synthesize V placeholder. For i4kv synthesise DequantV to match
		// the inference path (path 3: DequantV f16 + path 4: fused K inline-decode).
		// For K+V TQ presets value was already set above by DequantKV, so this
		// block is a no-op for those. For K-only presets value comes from Causal.
		// If the K block placed a K+V fused tensor (value=nil, V carried inline),
		// skip this entire block — otherwise we reintroduce the f16 V scratch.
		skipVPlaceholder := c.reserveSkipVPlaceholder
		c.reserveSkipVPlaceholder = false
		if !skipVPlaceholder && value == nil && c.headDim > 0 {
			nCells := 1
			if c.meta.curMask != nil {
				nCells = c.meta.curMask.Dim(0)
			}
			if value == nil {
				value = ctx.Input().Zeros(ml.DTypeF16, c.headDim, c.numKVHeads, nCells)
			}
		}
		return key, value, mask
	}

	if c.compressedK != nil {
		layer := c.meta.curLayer
		firstCell := c.meta.curCellRange.min
		nCells := c.meta.curMask.Dim(0)

		encodeResult := c.encodeResults[layer]
		vEncodeResult := c.vEncodeResults[layer]

		// 0. K-only symmetric presets (tq2k/tq3k/tq4k): fused K-only (no scratch).
		//    Falls back to DequantK only when fused is unavailable (D!=128).
		if c.preset.ValueBits == 0 && !c.preset.AsymmetricPrimary && encodeResult != nil {
			if c.fusedFallbackEligible {
				if tqk, ok := c.compressedK.GetAsTQTensor(ctx, layer, encodeResult, firstCell, nCells); ok {
					c.logPathOnce[0].Do(func() {
						slog.Info("turboquant: using K-only fused path")
					})
					_, value, mask := c.meta.Get(ctx)
					c.armRotationForNextSDPA()
					return tqk, value, mask
				}
			}
			key := c.compressedK.DequantK(ctx, layer, encodeResult, firstCell, nCells)
			if key != nil {
				c.logPathOnce[0].Do(func() {
					slog.Info("turboquant: using K-only DequantK + f16 V path (fused unavailable)")
				})
				_, value, mask := c.meta.Get(ctx)
				c.armRotationForNextSDPA()
				return key, value, mask
			}
		}

		// Prefill vs decode routing on Metal: the fused inline-decode kernel
		// re-decodes each packed K/V cell per Q-token, so its cost scales
		// O(nCells × nTokensQ). For prefill (nTokensQ ≫ 1) DequantKV + stock
		// FA wins by decoding each cell once and then letting stock FA
		// amortise the read across all Q-tokens. For decode (nTokensQ=1) the
		// fused path wins by avoiding the f16 intermediate write. This flag
		// routes independently of preferFusedAttn so CUDA (preferFusedAttn=
		// false) continues to take path 1 for everything.
		useDequantKVForPrefill := c.preferFusedAttn && c.curQueryLen > 1

		// 1. Combined K+V dequant → stock FA.
		//    Metal prefill only (batched Q decodes each K cell once; stock FA
		//    amortises the read across Q-tokens, beating fused for large batches).
		//    CUDA always uses path 2 (fused) to avoid materialising f16 scratch.
		if vEncodeResult != nil && useDequantKVForPrefill {
			key, value := c.compressedK.DequantKV(ctx, layer, encodeResult, vEncodeResult, firstCell, nCells)
			if key != nil && value != nil {
				c.logPathOnce[1].Do(func() {
					if useDequantKVForPrefill {
						slog.Info("turboquant: using combined DequantKV + stock FA path (Metal prefill)")
					} else {
						slog.Info("turboquant: using combined DequantKV + stock FA path")
					}
				})
				_, _, mask := c.meta.Get(ctx)
				c.armRotationForNextSDPA()
				return key, value, mask
			}
		}

		// 2. K+V fused inline-decode: reads packed K+V directly, no f16 intermediate.
		//    Primary path on CUDA/ROCm and Metal decode (Q=1).
		//    Instantiated at D=128 always; D=256 on Metal only.
		if vEncodeResult != nil && c.fusedFallbackEligible {
			if tqkv, ok := c.compressedK.GetAsTQTensorKV(ctx, layer, encodeResult, vEncodeResult, firstCell, nCells); ok {
				c.logPathOnce[2].Do(func() {
					if c.preferFusedAttn {
						slog.Info("turboquant: using K+V fused inline-decode path (Metal decode)")
					} else {
						slog.Info("turboquant: using K+V fused inline-decode path (CUDA)")
					}
				})
				_, _, mask := c.meta.Get(ctx)
				c.armRotationForNextSDPA()
				return tqkv, nil, mask
			}
		}

		// 1b. DequantKV fallback when Metal decode tried path 2 and the fused
		//     path was unavailable (e.g. headDim outside 128/256).
		if vEncodeResult != nil && c.preferFusedAttn {
			key, value := c.compressedK.DequantKV(ctx, layer, encodeResult, vEncodeResult, firstCell, nCells)
			if key != nil && value != nil {
				c.logPathOnce[1].Do(func() {
					slog.Info("turboquant: using combined DequantKV + stock FA path (fused unavailable)")
				})
				_, _, mask := c.meta.Get(ctx)
				c.armRotationForNextSDPA()
				return key, value, mask
			}
		}

		// 3. V-only dequant for K-only fused or separate K dequant fallback.
		var value ml.Tensor
		if vEncodeResult != nil {
			value = c.compressedK.DequantV(ctx, layer, vEncodeResult, firstCell, nCells)
		}

		// 4. Try K-only fused: K decoded inline, V is dequanted f16. Gated on
		//    fusedFallbackEligible for the same D=128 reason as path 2.
		if c.fusedFallbackEligible {
			if tqk, ok := c.compressedK.GetAsTQTensor(ctx, layer, encodeResult, firstCell, nCells); ok {
				c.logPathOnce[3].Do(func() {
					slog.Warn("turboquant: falling back to K-only inline-decode fused kernel")
				})
				_, metaValue, mask := c.meta.Get(ctx)
				if value == nil {
					value = metaValue
				}
				c.armRotationForNextSDPA()
				return tqk, value, mask
			}
		}

		// 5. Separate K + V dequant fallback (last resort).
		c.logPathOnce[4].Do(func() {
			slog.Warn("turboquant: falling back to separate K + V dequant path")
		})
		key := c.compressedK.DequantK(ctx, layer, encodeResult, firstCell, nCells)
		_, metaValue, mask := c.meta.Get(ctx)
		if value == nil {
			value = metaValue
		}
		c.armRotationForNextSDPA()
		return key, value, mask
	}

	return c.meta.Get(ctx)
}

// armRotationForNextSDPA sets the backend's tqRotationMatrix (and
// tqVRotationMatrix for K+V presets) so the next SDPA call — which happens
// immediately after this Get returns — rotates Q to match the TQ-rotated K
// and applies the V rotation undo after attention. SDPA reads-and-clears the
// flags, so non-TQ sub-cache layers in a WrapperCache (e.g. gemma3 SWA
// layers) are unaffected.
func (c *TurboQuantCache) armRotationForNextSDPA() {
	if c.rotMatrix == nil || c.rotSetter == nil {
		return
	}
	c.rotSetter.SetTQRotationMatrix(c.rotMatrix)
	// For K+V presets (tq3/tq2), also arm the V rotation undo on the next
	// SDPA call. For K-only presets (tq3k/tq2k), c.vRotMatrix is nil so the
	// V rotation is not armed and SDPA's consumed vRot stays nil.
	if c.vRotMatrix != nil {
		c.rotSetter.SetTQVRotationMatrix(c.vRotMatrix)
	}
}

// activateGPUEncode initialises the TQ compressed-K manager if the backend
// supports it and re-enables Q rotation (stored K is in rotated space).
func (c *TurboQuantCache) activateGPUEncode() {
	// fallbackToF16 un-skips K/V on the inner Causal so subsequent Put/Get
	// on this cache store and read f16 tensors like an ordinary non-TQ cache.
	// Init() sets SkipK/SkipV unconditionally, so any failure to activate GPU
	// encode must reverse those flags before the current Put() continues into
	// c.meta.Put — otherwise SDPA receives a nil K/V from Get and segfaults.
	// K allocation in Causal.Put is lazy (keyed on presence of c.keys[layer])
	// so flipping the flag on the first Put is sufficient.
	fallbackToF16 := func() {
		c.meta.SkipK = false
		c.meta.SkipV = false
	}

	// QJL residual is only algorithmically active when the preset also uses
	// outlier split (see turboquant.encodeVectorWithOutliers). The shipped
	// presets tq2/tq3/tq2k/tq3k have QJLRowsDivisor=1 at the algorithm
	// layer but outlierCount=0 — i.e. they do NOT use QJL at runtime.
	// Passing a nonzero qjlRows to the GPU manager in those cases would
	// trigger the "measurement-grade, needs CUDA+outliers" guard and cause
	// a silent f16 fallback. Gate the effective runtime qjlRows on
	// HasOutlierSplit to match the algorithm layer.
	qjlRows := 0
	if c.preset.HasOutlierSplit() {
		qjlRows = c.preset.KeyQJLRows(c.headDim)
	}
	// The user requested a measurement-grade preset if either asymmetric
	// primary quantization or the QJL residual sketch is on. If we can't
	// activate those on the GPU we produce exactly f16 output and any PPL
	// measurement is a lie. Log that failure at ERROR with a distinctive
	// marker so smoke tests and humans can both notice instead of silently
	// reading bit-identical-to-f16 numbers and calling them success.
	measurementRequested := c.preset.AsymmetricPrimary || qjlRows > 0
	logActivation := func(gpuActive bool, path string) {
		level := slog.LevelInfo
		if measurementRequested && !gpuActive {
			level = slog.LevelError
		}
		slog.Log(nil, level, "TQ_ACTIVATION",
			"preset", c.preset.Name,
			"asymmetric_requested", c.preset.AsymmetricPrimary,
			"qjl_rows_requested", qjlRows,
			"outlier_count", c.preset.OutlierCount,
			"gpu_active", gpuActive,
			"measurement_requested", measurementRequested,
			"path", path,
		)
	}

	tqb, ok := c.meta.backend.(ml.TQCompressedKBackend)
	if !ok {
		logActivation(false, "f16-fallback-no-tq-backend")
		fallbackToF16()
		return
	}

	// Pass the preset's outlier config so the manager can enable post-rotation
	// outlier split on the GPU encode path. This is required for correct
	// output on models with learned K bias (e.g. qwen2 family) and matches
	// the TurboQuant paper's validated experimental setup.
	mgr := tqb.NewTQCompressedKManager(
		c.headDim, c.numKVHeads, c.preset.KeyPrimaryBits, c.preset.RotationSeed,
		c.preset.ValueBits, c.preset.OutlierBits, c.preset.OutlierCount,
		c.preset.AsymmetricPrimary, qjlRows,
	)
	if mgr == nil {
		if measurementRequested {
			slog.Error("turboquant: MEASUREMENT PRESET REQUESTED BUT GPU UNAVAILABLE — SILENT F16 FALLBACK ACTIVE. Set OLLAMA_KV_CACHE_TYPE to a non-measurement preset (tq3k/tq3/tq2k/tq2) or run on a CUDA GPU.",
				"preset", c.preset.Name,
				"asymmetric", c.preset.AsymmetricPrimary,
				"qjl_rows", qjlRows)
		} else {
			slog.Info("turboquant: GPU encode not available, using f16 K fallback")
		}
		logActivation(false, "f16-fallback-mgr-nil")
		fallbackToF16()
		return
	}
	c.compressedK = mgr
	// Apply any K projection biases buffered before the manager was ready.
	if len(c.pendingKBiases) > 0 {
		type kBiasSetter interface {
			SetLayerKBias(layer int, bias ml.Tensor)
		}
		if kbs, ok := mgr.(kBiasSetter); ok {
			for layer, bias := range c.pendingKBiases {
				kbs.SetLayerKBias(layer, bias)
			}
		}
		c.pendingKBiases = nil
	}
	logActivation(true, "gpu-native")

	type fusedAttnPreferrer interface {
		PreferFusedAttention() bool
	}
	if ff, ok := mgr.(fusedAttnPreferrer); ok {
		c.preferFusedAttn = ff.PreferFusedAttention()
		if c.preferFusedAttn {
			slog.Info("turboquant: preferring fused flash-attention path (Metal: avoids f16 intermediate buffer)")
		}
	}

	// The inline-decode fused-FA fallback paths (Get paths 2 and 4) dispatch
	// to a kernel that is D-specialised. CUDA has only D=128 today; Metal has
	// D=128 and D=256 (kernel_tq_fattn_vec_*{,_d256}). Models with an
	// unsupported head dim (e.g. gemma4 D=512, or gemma3 D=256 on CUDA) must
	// skip these fallbacks to avoid a kernel-side GGML_ASSERT; path 5
	// (separate K+V dequant) handles them correctly.
	c.fusedFallbackEligible = c.headDim == 128 ||
		(c.headDim == 256 && c.preferFusedAttn)
	if !c.fusedFallbackEligible {
		reason := "headDim != 128"
		if c.headDim == 256 {
			reason = "headDim == 256 but backend lacks D=256 fused kernel"
		}
		slog.Info("turboquant: inline-decode fused-FA fallback paths disabled",
			"reason", reason, "headDim", c.headDim)
	}

	// Cache the rotation matrices and the backend's rotation-setter hook on
	// TurboQuantCache so Get() can arm them per-call without re-running a
	// type assertion every layer. We do NOT set them at activate time — a
	// sticky backend-global rotation would corrupt attention on unwrapped
	// SWA layers in mixed-head-dim models like gemma3.
	c.rotMatrix = mgr.RotationMatrix(nil, 0)
	if rs, ok := c.meta.backend.(tqRotSetter); ok {
		c.rotSetter = rs
	}

	type vRotFusedSetter interface {
		SetTQVRotFusedInDequant(bool)
	}
	type vRotProvider interface {
		RotationMatrixR() ml.Tensor
	}
	if c.preset.ValueBits > 0 {
		if vp, ok := mgr.(vRotProvider); ok {
			c.vRotMatrix = vp.RotationMatrixR()
		}
		// DequantKV outputs R^T @ v (still rotated). SDPA applies the rotation
		// undo as R @ attn_out via mulmat, which is dramatically faster than
		// the per-cell matmul of the fused dequant kernel.
		if rs, ok := c.meta.backend.(vRotFusedSetter); ok {
			rs.SetTQVRotFusedInDequant(false)
		}
	}

	slog.Info("turboquant: GPU-native encode active",
		"headDim", c.headDim, "numKVHeads", c.numKVHeads,
		"K_bits", c.preset.KeyPrimaryBits, "V_bits", c.preset.ValueBits)
}

func (c *TurboQuantCache) CopyPrefix(srcSeq, dstSeq int, prefixLen int32) {
	c.meta.CopyPrefix(srcSeq, dstSeq, prefixLen)
}

func (c *TurboQuantCache) CanResume(seq int, pos int32) bool {
	return c.meta.CanResume(seq, pos)
}

// Remove returns ErrNotSupported for any partial eviction when GPU-compressed
// K is active: the compressed buffer cannot be RoPE-shifted in-place, and the
// inner Causal.Remove path silently skips its shiftFn loop because SkipK
// leaves c.keys empty — survivors would keep their old RoPE embeddings while
// their positions are decremented, producing wrong attention scores. Only a
// full-sequence eviction (Remove(seq, 0, MaxInt32)) avoids the shift path;
// other callers must fall back to full reprocessing via ErrReprocessInputs.
func (c *TurboQuantCache) Remove(seq int, beginIndex, endIndex int32) error {
	if c.compressedK != nil && !(beginIndex == 0 && endIndex == math.MaxInt32) {
		return ErrNotSupported
	}
	return c.meta.Remove(seq, beginIndex, endIndex)
}

func (c *TurboQuantCache) SetCausal(ctx ml.Context, opts CausalOptions) {
	c.meta.SetCausal(ctx, opts)
}

func PresetFromDType(dtype ml.DType) (turboquant.Preset, bool) {
	switch dtype {
	case ml.DTypeTQ2:
		return turboquant.PresetTQ2, true
	case ml.DTypeTQ3:
		return turboquant.PresetTQ3, true
	case ml.DTypeTQ3K:
		return turboquant.PresetTQ3K, true
	case ml.DTypeTQ2K:
		return turboquant.PresetTQ2K, true
	case ml.DTypeTQ3A:
		return turboquant.PresetTQ3A, true
	case ml.DTypeTQ3KA:
		return turboquant.PresetTQ3KA, true
	case ml.DTypeTQ2A:
		return turboquant.PresetTQ2A, true
	case ml.DTypeTQ2KA:
		return turboquant.PresetTQ2KA, true
	case ml.DTypeTQ3QA:
		return turboquant.PresetTQ3QA, true
	case ml.DTypeTQ2QA:
		return turboquant.PresetTQ2QA, true
	case ml.DTypeTQ4:
		return turboquant.PresetTQ4, true
	case ml.DTypeTQ4K:
		return turboquant.PresetTQ4K, true
	case ml.DTypeTQ4A:
		return turboquant.PresetTQ4A, true
	case ml.DTypeTQ4KA:
		return turboquant.PresetTQ4KA, true
	case ml.DTypeTQ4QA:
		return turboquant.PresetTQ4QA, true
	default:
		return turboquant.Preset{}, false
	}
}

var _ Cache = (*TurboQuantCache)(nil)

// Note: TurboQuantCache intentionally does NOT implement CheckpointCache.
// CheckpointCache is for recurrent caches that need special per-sequence
// state restoration. The inner *Causal cache supports plain CanResume, which
// is the right semantics for TQ-wrapped caches too — the runner will fall
// through to the CanResume branch when TurboQuantCache is in use (see
// runner/ollamarunner/cache.go:163-170). An earlier implementation had a
// stub PrepareRestore that always returned (0, false), which forced a full
// prompt reprocess on every resume for long-context gemma/llama runs.
