// kdump diagnoses TurboQuant codebook calibration by comparing the empirical
// distribution of post-rotation K coordinates against the theoretical
// unit-vector marginal distribution that the Lloyd-Max codebook is built from.
//
// Usage:
//
//	go build -o dist/bin/kdump ./cmd/kdump
//
//	cat wikitext-2-raw/wiki.test.raw | \
//	  OLLAMA_LIBRARY_PATH=./build/lib/ollama \
//	  ./dist/bin/kdump --model path/to/model.gguf --preset tq3 --ctx 256 \
//	                   --out /tmp/kdump.csv
//
// The tool runs a single forward pass over --ctx tokens, reads K tensors from
// the f32 KV cache (no quantization), applies the TurboQuant Householder
// rotation, and compares the resulting coordinate distribution to the
// theoretical distribution used to build the Lloyd-Max codebook.
//
// Output (stdout) is a CSV with columns: source,layer,value
//   - source: "empirical" (from model) or "theoretical" (codebook calibration)
//   - layer: layer index for empirical, -1 for theoretical
//   - value: rotated coordinate
//
// Summary statistics are printed to stderr.
//
// Quick Python analysis after capture:
//
//	import pandas as pd, matplotlib.pyplot as plt
//	df = pd.read_csv('/tmp/kdump.csv')
//	df.groupby('source')['value'].describe()
//	df[df.source=='empirical'].hist(bins=100, density=True)
//	df[df.source=='theoretical'].hist(bins=100, density=True)
//	plt.show()
package main

import (
	"bufio"
	"context"
	"encoding/csv"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"runtime"
	"slices"
	"strconv"
	"strings"

	ggmlbackend "github.com/ollama/ollama/ml/backend/ggml"

	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
	_ "github.com/ollama/ollama/model/models"
	"github.com/ollama/ollama/tokenizer"
	"github.com/ollama/ollama/turboquant"
)

func main() {
	modelPath := flag.String("model", "", "path to GGUF model file (required)")
	presetName := flag.String("preset", "tq3", "TQ preset whose rotation seed to use (tq2, tq3, tq3k, …)")
	ctxLen := flag.Int("ctx", 256, "tokens to process")
	gpuLayers := flag.Int("gpu-layers", 200, "layers to offload to GPU")
	inputFile := flag.String("input", "", "input text file (default: stdin)")
	outFile := flag.String("out", "/tmp/kdump.csv", "output CSV path")
	nTheory := flag.Int("theory-samples", 65536, "number of theoretical samples per layer")
	flag.Parse()

	if *modelPath == "" {
		log.Fatal("--model is required")
	}

	preset, err := turboquant.PresetByName(*presetName)
	if err != nil {
		log.Fatalf("unknown preset %q: %v", *presetName, err)
	}

	params := ml.BackendParams{
		AllocMemory:    true,
		NumThreads:     runtime.NumCPU(),
		FlashAttention: ml.FlashAttentionEnabled,
	}
	if *gpuLayers > 0 {
		gpus := ggmlbackend.GPUDevices()
		if len(gpus) > 0 {
			layers := make([]int, *gpuLayers)
			for i := range layers {
				layers[i] = i
			}
			params.GPULayers = ml.GPULayersList{{
				DeviceID: ml.DeviceID{ID: gpus[0].ID, Library: gpus[0].Library},
				Layers:   layers,
			}}
		} else {
			log.Println("warning: no GPU devices found; running on CPU")
		}
	}

	m, err := model.New(*modelPath, params)
	if err != nil {
		log.Fatalf("load model: %v", err)
	}
	fmt.Fprint(os.Stderr, "loading weights... ")
	if err := m.Backend().Load(context.TODO(), nil); err != nil {
		log.Fatalf("load weights: %v", err)
	}
	fmt.Fprintln(os.Stderr, "done")

	// Use f32 storage so Floats() reads correctly (no f16→f32 conversion issue).
	causal, ok := m.Config().Cache.(*kvcache.Causal)
	if !ok {
		log.Fatal("model cache is not *kvcache.Causal — unsupported model type")
	}
	causal.DTypeK = ml.DTypeF32
	causal.DTypeV = ml.DTypeF32
	causal.Init(m.Backend(), ml.DTypeF32, 1, *ctxLen, *ctxLen)

	tok, ok2 := m.(tokenizer.Tokenizer)
	if !ok2 {
		log.Fatal("model does not implement tokenizer interface")
	}

	// Read text.
	var r *os.File = os.Stdin
	if *inputFile != "" {
		f, err := os.Open(*inputFile)
		if err != nil {
			log.Fatal(err)
		}
		defer f.Close()
		r = f
	}
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 64*1024*1024), 64*1024*1024)
	var sb strings.Builder
	for scanner.Scan() {
		sb.WriteString(scanner.Text())
		sb.WriteByte('\n')
	}
	text := sb.String()

	tokens, err := tok.Encode(text, false)
	if err != nil {
		log.Fatalf("tokenize: %v", err)
	}

	n := min(*ctxLen, len(tokens))
	if n < 2 {
		log.Fatal("not enough tokens")
	}
	chunk := tokens[:n]

	fmt.Fprintf(os.Stderr, "preset=%s  rotation_seed=0x%x  tokens=%d\n",
		preset.Name, preset.RotationSeed, n)

	// Forward pass — only request a single output position so we don't waste
	// memory, but we need all n tokens put through the cache to populate K.
	inToks := make([]int32, n)
	for i, t := range chunk {
		inToks[i] = int32(t)
	}
	outPos := []int32{int32(n - 1)}

	ctx := m.Backend().NewContext()
	ctx.SetBatchSize(n)

	batch := input.Batch{
		Positions: make([]int32, n),
		Sequences: make([]int, n),
	}
	for i := range batch.Positions {
		batch.Positions[i] = int32(i)
	}
	batch.Inputs = ctx.Input().FromInts(inToks, n)
	batch.Outputs = ctx.Input().FromInts(outPos, 1)

	logitTensor, err := model.Forward(ctx, m, batch)
	if err != nil {
		log.Fatalf("forward: %v", err)
	}
	ctx.Compute(logitTensor)

	// Read K tensors from the persistent cache storage.
	// Shape: [kHeadDim, numKVHeads, cacheCapacity] stored as f32.
	keys := causal.Keys()
	if len(keys) == 0 {
		log.Fatal("no K tensors in cache — forward pass did not populate cache")
	}

	// Sort layer indices for deterministic output.
	layerIdx := make([]int, 0, len(keys))
	for l := range keys {
		layerIdx = append(layerIdx, l)
	}
	slices.Sort(layerIdx)

	firstLayer := layerIdx[0]
	headDim := keys[firstLayer].Dim(0)
	numKVHeads := keys[firstLayer].Dim(1)

	fmt.Fprintf(os.Stderr, "layers=%d  kHeadDim=%d  numKVHeads=%d\n",
		len(layerIdx), headDim, numKVHeads)

	rotation := turboquant.BuildRotation(headDim, preset.RotationSeed)

	// Collect empirical rotated coordinates per layer.
	type stat struct {
		n, sum, sumSq, sumCubed, sumFourth float64
		min, max                            float64
	}
	accumStat := func(s *stat, v float64) {
		s.n++
		s.sum += v
		s.sumSq += v * v
		s.sumCubed += v * v * v
		s.sumFourth += v * v * v * v
		if v < s.min {
			s.min = v
		}
		if v > s.max {
			s.max = v
		}
	}
	finishStat := func(s stat) (mean, std, kurt float64) {
		if s.n == 0 {
			return 0, 0, 0
		}
		mean = s.sum / s.n
		variance := s.sumSq/s.n - mean*mean
		if variance < 0 {
			variance = 0
		}
		std = math.Sqrt(variance)
		if std == 0 {
			return mean, 0, 0
		}
		// Excess kurtosis: E[(x-μ)^4]/σ^4 - 3
		m4 := s.sumFourth/s.n - 4*mean*s.sumCubed/s.n + 6*mean*mean*s.sumSq/s.n - 3*mean*mean*mean*mean
		kurt = m4/(variance*variance) - 3
		return mean, std, kurt
	}

	// Open CSV output.
	out, err := os.Create(*outFile)
	if err != nil {
		log.Fatalf("create output: %v", err)
	}
	defer out.Close()
	w := csv.NewWriter(bufio.NewWriter(out))
	_ = w.Write([]string{"source", "layer", "value"})

	var globalEmp stat
	globalEmp.min = math.MaxFloat64
	globalEmp.max = -math.MaxFloat64

	layerStr := func(l int) string { return strconv.Itoa(l) }

	for _, layer := range layerIdx {
		kt := keys[layer]
		allK := kt.BackendGet()
		if len(allK) == 0 {
			fmt.Fprintf(os.Stderr, "layer %d: BackendGet() returned empty slice\n", layer)
			continue
		}

		// allK layout: [d, h, cell] → index = d + h*headDim + cell*headDim*numKVHeads
		// We only process the first n cells (positions 0..n-1 written this pass).
		headVec := make([]float32, headDim)
		var ls stat
		ls.min = math.MaxFloat64
		ls.max = -math.MaxFloat64
		ls.n = 0
		ls.sum = 0
		ls.sumSq = 0
		ls.sumCubed = 0
		ls.sumFourth = 0

		lyr := layerStr(layer)
		for cell := range n {
			for head := range numKVHeads {
				base := cell*headDim*numKVHeads + head*headDim
				if base+headDim > len(allK) {
					fmt.Fprintf(os.Stderr, "layer %d cell %d head %d: out of bounds (len=%d)\n",
						layer, cell, head, len(allK))
					continue
				}
				copy(headVec, allK[base:base+headDim])
				rotated := turboquant.ApplyRotation(headVec, rotation)
				// Normalize by RMS scale — mirrors blockScale in encode.go.
				// The codebook operates on normalized = rotated / scale, so
				// that is the distribution we must compare against theoretical.
				var sumSq float64
				for _, v := range rotated {
					sumSq += float64(v) * float64(v)
				}
				scale := math.Sqrt(sumSq / float64(headDim))
				if scale < 1e-7 {
					continue // skip near-zero head vectors
				}
				for _, v := range rotated {
					fv := float64(v) / scale
					accumStat(&ls, fv)
					accumStat(&globalEmp, fv)
					_ = w.Write([]string{"empirical", lyr, strconv.FormatFloat(fv, 'f', 6, 64)})
				}
			}
		}

		mean, std, kurt := finishStat(ls)
		fmt.Fprintf(os.Stderr, "  layer %2d: n=%-9.0f  mean=%+.4f  std=%.4f  kurt=%+.4f  [%.3f, %.3f]\n",
			layer, ls.n, mean, std, kurt, ls.min, ls.max)
	}

	ctx.Close()

	// Generate theoretical samples.
	fmt.Fprintf(os.Stderr, "\ngenerating %d theoretical samples (dim=%d bits=%d)...\n",
		*nTheory, headDim, preset.KeyPrimaryBits)
	theory := turboquant.UnitVectorCoordSamples(headDim, preset.KeyPrimaryBits, *nTheory)

	var theoryStat stat
	theoryStat.min = math.MaxFloat64
	theoryStat.max = -math.MaxFloat64
	for _, v := range theory {
		accumStat(&theoryStat, v)
		_ = w.Write([]string{"theoretical", "-1", strconv.FormatFloat(v, 'f', 6, 64)})
	}

	w.Flush()
	if err := w.Error(); err != nil {
		log.Fatalf("csv flush: %v", err)
	}

	// Print summary comparison.
	empMean, empStd, empKurt := finishStat(globalEmp)
	thMean, thStd, thKurt := finishStat(theoryStat)

	fmt.Fprintf(os.Stderr, "\n=== Distribution comparison ===\n")
	fmt.Fprintf(os.Stderr, "%-20s  %10s  %10s  %10s  %10s  %10s\n",
		"source", "n", "mean", "std", "kurt(excess)", "range")
	fmt.Fprintf(os.Stderr, "%-20s  %10.0f  %+10.4f  %10.4f  %+10.4f  [%.3f, %.3f]\n",
		"empirical", globalEmp.n, empMean, empStd, empKurt, globalEmp.min, globalEmp.max)
	fmt.Fprintf(os.Stderr, "%-20s  %10.0f  %+10.4f  %10.4f  %+10.4f  [%.3f, %.3f]\n",
		"theoretical", theoryStat.n, thMean, thStd, thKurt, theoryStat.min, theoryStat.max)

	// KS statistic: sort both, merge-walk CDFs.
	slices.Sort(theory)
	empAll := make([]float64, 0, int(globalEmp.n))
	// Re-read from file is expensive; recompute from keys in memory.
	for _, layer := range layerIdx {
		kt := keys[layer]
		allK := kt.BackendGet()
		hv := make([]float32, headDim)
		for cell := range n {
			for head := range numKVHeads {
				base := cell*headDim*numKVHeads + head*headDim
				if base+headDim > len(allK) {
					continue
				}
				copy(hv, allK[base:base+headDim])
				rotated := turboquant.ApplyRotation(hv, rotation)
				var sumSq float64
				for _, v := range rotated {
					sumSq += float64(v) * float64(v)
				}
				scale := math.Sqrt(sumSq / float64(headDim))
				if scale < 1e-7 {
					continue
				}
				for _, v := range rotated {
					empAll = append(empAll, float64(v)/scale)
				}
			}
		}
	}
	slices.Sort(empAll)

	ksD := ksDistance(empAll, theory)
	fmt.Fprintf(os.Stderr, "\nKolmogorov-Smirnov distance: %.4f\n", ksD)
	if ksD < 0.05 {
		fmt.Fprintln(os.Stderr, "→ distributions match well; codebook calibration appears correct")
	} else if ksD < 0.15 {
		fmt.Fprintln(os.Stderr, "→ moderate mismatch; codebook may be slightly miscalibrated")
	} else {
		fmt.Fprintln(os.Stderr, "→ SIGNIFICANT MISMATCH; codebook is miscalibrated for this model's K distribution")
		fmt.Fprintln(os.Stderr, "  Recalibrate by running Lloyd-Max on empirical samples (see turboquant/codebook.go)")
	}

	// Quantile comparison.
	fmt.Fprintf(os.Stderr, "\n%-8s  %8s  %8s  %8s\n", "quantile", "empirical", "theory", "diff")
	for _, q := range []float64{0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99} {
		eq := quantile(empAll, q)
		tq := quantile(theory, q)
		fmt.Fprintf(os.Stderr, "q%-7.2f  %8.4f  %8.4f  %+8.4f\n", q, eq, tq, eq-tq)
	}

	fmt.Fprintf(os.Stderr, "\nCSV written to %s\n", *outFile)
}

// ksDistance computes the Kolmogorov-Smirnov statistic between two pre-sorted
// samples: max |F_a(x) - F_b(x)| over all x.
func ksDistance(a, b []float64) float64 {
	na, nb := len(a), len(b)
	if na == 0 || nb == 0 {
		return 1
	}
	ia, ib := 0, 0
	maxD := 0.0
	for ia < na || ib < nb {
		var x float64
		if ia < na && (ib >= nb || a[ia] <= b[ib]) {
			x = a[ia]
		} else {
			x = b[ib]
		}
		// Advance both past x.
		for ia < na && a[ia] <= x {
			ia++
		}
		for ib < nb && b[ib] <= x {
			ib++
		}
		fa := float64(ia) / float64(na)
		fb := float64(ib) / float64(nb)
		d := math.Abs(fa - fb)
		if d > maxD {
			maxD = d
		}
	}
	return maxD
}

// quantile returns the p-th quantile of a pre-sorted slice.
func quantile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return math.NaN()
	}
	idx := p * float64(len(sorted)-1)
	lo := int(idx)
	hi := lo + 1
	if hi >= len(sorted) {
		return sorted[len(sorted)-1]
	}
	frac := idx - float64(lo)
	return sorted[lo]*(1-frac) + sorted[hi]*frac
}
