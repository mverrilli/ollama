package ggml

// #include "ggml/include/ggml.h"
import "C"

import (
	"log/slog"
	"sync"

	ml "github.com/ollama/ollama/ml"
)

// ggmlQ8KCompressedK implements ml.Q8KCompressedKManager using ggml tensors
// and GGML_OP_Q8K_ENCODE / GGML_OP_Q8K_DEQUANT (or Q4K variants) ops. All
// buffers are GPU-resident; no CPU round-trips occur during inference.
//
// Storage layout per layer (group size g=32):
//   K packed:  [headDim * numKVHeads, capacity] i8  (int8) or
//              [headDim/2 * numKVHeads, capacity] i8 (int4, nibble-packed)
//   K scales:  [(headDim/g) * numKVHeads, capacity] f16
//   K mins:    [(headDim/g) * numKVHeads, capacity] f16
//   V packed/scales/mins: same layout as K (when withV is true)
type ggmlQ8KCompressedK struct {
	backend    *Backend
	headDim    int
	numKVHeads int
	is4Bit     bool // false = int8 (q8k), true = int4/nibble (q4k)
	withV      bool // true when V is also compressed (q8kv/q4kv)

	mu sync.Mutex

	// Per-layer K tensors, allocated lazily via EnsureLayer.
	layerCtxs    map[int]ml.Context
	kPackedTensors map[int]*Tensor // K packed
	kScalesTensors map[int]*Tensor // K per-group scales [f16]
	kMinsTensors   map[int]*Tensor // K per-group mins   [f16]

	// Per-layer V tensors, allocated lazily via EnsureVLayer.
	vLayerCtxs    map[int]ml.Context
	vPackedTensors map[int]*Tensor
	vScalesTensors map[int]*Tensor
	vMinsTensors   map[int]*Tensor
}

// kPackedStride returns the number of bytes per (head × cell) in the packed K buffer.
// For int8: headDim bytes; for int4: headDim/2 bytes (two nibbles per byte).
func (m *ggmlQ8KCompressedK) kPackedStride() int {
	if m.is4Bit {
		return m.headDim / 2
	}
	return m.headDim
}

// numGroups returns the number of per-group scale/min entries per head.
// Group size is always 32.
func (m *ggmlQ8KCompressedK) numGroups() int {
	return m.headDim / 32
}

// EnsureLayer allocates per-layer K tensors on first use.
func (m *ggmlQ8KCompressedK) EnsureLayer(layer, capacity int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.kPackedTensors[layer] != nil {
		return
	}
	// Use newTQContext so tensors land on the GPU buffer type.
	// 3 tensors per layer (packed, scales, mins).
	ctx := m.backend.newTQContext(3)
	ctx.layer = layer

	stride := m.kPackedStride()
	nGroups := m.numGroups()

	packed := ctx.Zeros(ml.DTypeI8, stride*m.numKVHeads, capacity).(*Tensor)
	scales := ctx.Zeros(ml.DTypeF16, nGroups*m.numKVHeads, capacity).(*Tensor)
	mins := ctx.Zeros(ml.DTypeF16, nGroups*m.numKVHeads, capacity).(*Tensor)

	m.layerCtxs[layer] = ctx
	m.kPackedTensors[layer] = packed
	m.kScalesTensors[layer] = scales
	m.kMinsTensors[layer] = mins
}

// EncodeK creates the encode graph node for K.
func (m *ggmlQ8KCompressedK) EncodeK(ctx ml.Context, layer int, key ml.Tensor, firstCell int) ml.Tensor {
	packed := m.kPackedTensors[layer]
	if packed == nil {
		return nil
	}
	scales := m.kScalesTensors[layer]
	mins := m.kMinsTensors[layer]
	if m.is4Bit {
		return packed.I4Q4Encode(ctx, scales, mins, key, firstCell)
	}
	return packed.I4Encode(ctx, scales, mins, key, firstCell)
}

// DequantK creates the dequant graph node for K, returning [headDim, numKVHeads, nCells] f16.
func (m *ggmlQ8KCompressedK) DequantK(ctx ml.Context, layer int, encodeResult ml.Tensor, firstCell, nCells int) ml.Tensor {
	if encodeResult == nil || nCells <= 0 {
		return nil
	}
	enc := encodeResult.(*Tensor)
	scales := m.kScalesTensors[layer]
	mins := m.kMinsTensors[layer]
	if scales == nil || mins == nil {
		return nil
	}
	if m.is4Bit {
		return enc.I4Q4Dequant(ctx, scales, mins, m.headDim, m.numKVHeads, nCells, firstCell)
	}
	return enc.I4Dequant(ctx, scales, mins, m.headDim, m.numKVHeads, nCells, firstCell)
}

// GetAsQ8KTensor returns a q8kTensor wrapper for the fused flash-attention
// path. Returns (nil, false) when the fused kernel is not supported.
func (m *ggmlQ8KCompressedK) GetAsQ8KTensor(ctx ml.Context, layer int, encodeResult ml.Tensor, firstCell, nCells int) (ml.Tensor, bool) {
	if encodeResult == nil || nCells <= 0 {
		return nil, false
	}
	// Fused kernel currently only available at headDim=128.
	if m.headDim != 128 {
		return nil, false
	}
	scales := m.kScalesTensors[layer]
	mins := m.kMinsTensors[layer]
	if scales == nil || mins == nil {
		return nil, false
	}
	return &q8kTensor{
		Tensor:    encodeResult.(*Tensor),
		scales:    scales,
		mins:      mins,
		is4Bit:    m.is4Bit,
		headDim:   m.headDim,
		nKVHeads:  m.numKVHeads,
		nCells:    nCells,
		firstCell: firstCell,
	}, true
}

// EnsureVLayer allocates per-layer V tensors on first use.
func (m *ggmlQ8KCompressedK) EnsureVLayer(layer, capacity int) {
	if !m.withV {
		return
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.vPackedTensors[layer] != nil {
		return
	}
	ctx := m.backend.newTQContext(3)
	ctx.layer = layer

	stride := m.kPackedStride()
	nGroups := m.numGroups()

	packed := ctx.Zeros(ml.DTypeI8, stride*m.numKVHeads, capacity).(*Tensor)
	scales := ctx.Zeros(ml.DTypeF16, nGroups*m.numKVHeads, capacity).(*Tensor)
	mins := ctx.Zeros(ml.DTypeF16, nGroups*m.numKVHeads, capacity).(*Tensor)

	m.vLayerCtxs[layer] = ctx
	m.vPackedTensors[layer] = packed
	m.vScalesTensors[layer] = scales
	m.vMinsTensors[layer] = mins
}

// EncodeV creates the encode graph node for V.
func (m *ggmlQ8KCompressedK) EncodeV(ctx ml.Context, layer int, value ml.Tensor, firstCell int) ml.Tensor {
	if !m.withV {
		return nil
	}
	packed := m.vPackedTensors[layer]
	if packed == nil {
		return nil
	}
	scales := m.vScalesTensors[layer]
	mins := m.vMinsTensors[layer]
	if m.is4Bit {
		return packed.I4Q4Encode(ctx, scales, mins, value, firstCell)
	}
	return packed.I4Encode(ctx, scales, mins, value, firstCell)
}

// DequantV creates the dequant graph node for V.
func (m *ggmlQ8KCompressedK) DequantV(ctx ml.Context, layer int, encodeResult ml.Tensor, firstCell, nCells int) ml.Tensor {
	if !m.withV || encodeResult == nil || nCells <= 0 {
		return nil
	}
	enc := encodeResult.(*Tensor)
	scales := m.vScalesTensors[layer]
	mins := m.vMinsTensors[layer]
	if scales == nil || mins == nil {
		return nil
	}
	if m.is4Bit {
		return enc.I4Q4Dequant(ctx, scales, mins, m.headDim, m.numKVHeads, nCells, firstCell)
	}
	return enc.I4Dequant(ctx, scales, mins, m.headDim, m.numKVHeads, nCells, firstCell)
}

// Close frees all GPU buffers by releasing the ggml contexts.
func (m *ggmlQ8KCompressedK) Close() {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, ctx := range m.layerCtxs {
		ctx.(*Context).Close()
	}
	m.layerCtxs = make(map[int]ml.Context)
	for _, ctx := range m.vLayerCtxs {
		ctx.(*Context).Close()
	}
	m.vLayerCtxs = make(map[int]ml.Context)
}

// NewQ8KCompressedKManager implements ml.Q8KCompressedKBackend.
func (b *Backend) NewQ8KCompressedKManager(headDim, numKVHeads int, is4Bit, withV bool) ml.Q8KCompressedKManager {
	scan := b.scanTQDevices()
	if !scan.selectedOK {
		if len(scan.Skipped) > 0 {
			slog.Warn("q8k: no GPU found; falling back to f16 KV cache",
				"skipped_gpus", scan.Skipped)
		} else {
			slog.Warn("q8k: no GPU backend available, falling back to f16 KV cache")
		}
		return nil
	}
	scheme := "q8k"
	if is4Bit {
		scheme = "q4k"
	}
	if withV {
		scheme += "v"
	}
	slog.Info("q8k: initializing per-group integer KV cache manager",
		"scheme", scheme, "headDim", headDim, "numKVHeads", numKVHeads)
	return &ggmlQ8KCompressedK{
		backend:        b,
		headDim:        headDim,
		numKVHeads:     numKVHeads,
		is4Bit:         is4Bit,
		withV:          withV,
		layerCtxs:      make(map[int]ml.Context),
		kPackedTensors: make(map[int]*Tensor),
		kScalesTensors: make(map[int]*Tensor),
		kMinsTensors:   make(map[int]*Tensor),
		vLayerCtxs:     make(map[int]ml.Context),
		vPackedTensors: make(map[int]*Tensor),
		vScalesTensors: make(map[int]*Tensor),
		vMinsTensors:   make(map[int]*Tensor),
	}
}

// Compile-time interface check.
var _ ml.Q8KCompressedKManager = (*ggmlQ8KCompressedK)(nil)
var _ ml.Q8KCompressedKBackend = (*Backend)(nil)
