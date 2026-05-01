package ggml

import ml "github.com/ollama/ollama/ml"

// q8kTensor wraps a packed-K buffer with scale/min metadata for the fused
// Q8K (or Q4K) flash-attention kernel. The inner Tensor is the encode result
// (a view of the persistent packed-K GPU buffer).
type q8kTensor struct {
	*Tensor           // packed K view ([stride*nKVHeads, capacity] i8)
	scales    *Tensor // K per-group scales [(headDim/32)*numKVHeads, capacity] f16
	mins      *Tensor // K per-group mins   [(headDim/32)*numKVHeads, capacity] f16
	is4Bit    bool    // true = nibble-packed (q4k); false = full-byte (q8k)
	headDim   int
	nKVHeads  int
	nCells    int
	firstCell int
}

// Permute propagates the q8kTensor wrapper through the key permutation that
// ScaledDotProductAttention applies before the flash-attention dispatch.
func (t *q8kTensor) Permute(ctx ml.Context, shape ...int) ml.Tensor {
	return &q8kTensor{
		Tensor:    t.Tensor.Permute(ctx, shape...).(*Tensor),
		scales:    t.scales,
		mins:      t.mins,
		is4Bit:    t.is4Bit,
		headDim:   t.headDim,
		nKVHeads:  t.nKVHeads,
		nCells:    t.nCells,
		firstCell: t.firstCell,
	}
}

// q8kFlashAttention dispatches to GGML_OP_Q8K_FLASH_ATTN_EXT or
// GGML_OP_Q4K_FLASH_ATTN_EXT depending on q8kTensor.is4Bit.
func (b *Backend) q8kFlashAttention(
	ctx ml.Context,
	query *Tensor,
	qk *q8kTensor,
	value *Tensor,
	mask ml.Tensor,
	scale float64,
	logitSoftcap float64,
) ml.Tensor {
	if qk.is4Bit {
		return qk.Tensor.Q4KFlashAttnExt(ctx, query, value, mask, qk.scales, qk.mins,
			float32(scale), float32(logitSoftcap), qk.firstCell, qk.nKVHeads, qk.nCells)
	}
	return qk.Tensor.Q8KFlashAttnExt(ctx, query, value, mask, qk.scales, qk.mins,
		float32(scale), float32(logitSoftcap), qk.firstCell, qk.nKVHeads, qk.nCells)
}
