// perplexity measures KV-cache-compression perplexity for a given preset,
// along with KV cache size, total VRAM, prefill tok/s, and decode tok/s.
//
// Usage (after cmake --build build -j 4 and go build):
//
//	go build -o dist/bin/perplexity ./cmd/perplexity
//
//	cat wikitext-2-raw/wiki.test.raw | \
//	  OLLAMA_LIBRARY_PATH=./build/lib/ollama \
//	  ./dist/bin/perplexity --model path/to/model.gguf --preset f16 --ctx 512
//
//	# compare presets:
//	for p in f16 q8_0 q8k q8kv q4k q4kv tq2 tq3 tq4 tq2k tq3k tq4k tq3qa tq2qa tq4qa; do
//	  cat text.txt | OLLAMA_LIBRARY_PATH=./build/lib/ollama \
//	    ./dist/bin/perplexity --model model.gguf --preset $p
//	done
//
// OLLAMA_LIBRARY_PATH must point to the directory containing libggml-cuda.so
// (typically build/lib/ollama after a cmake build).
//
// VRAM is sampled via nvidia-smi at three points: before model load, after
// model load (weights on GPU), and after the first forward pass (compute
// graph reserved). The last delta captures the full TQ vs f16 difference.
//
// Flash attention is enabled by default (--flash-attention=true).
package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"time"

	ggmlbackend "github.com/ollama/ollama/ml/backend/ggml"

	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
	_ "github.com/ollama/ollama/model/models" // registers all model architectures
	"github.com/ollama/ollama/tokenizer"
)

// kvDTypeFromStr mirrors the private kvCacheTypeFromStr in ollamarunner.
func kvDTypeFromStr(s string) ml.DType {
	switch s {
	case "q8_0":
		return ml.DTypeQ80
	case "q4_0":
		return ml.DTypeQ40
	case "tq2":
		return ml.DTypeTQ2
	case "tq3":
		return ml.DTypeTQ3
	case "tq3k":
		return ml.DTypeTQ3K
	case "tq2k":
		return ml.DTypeTQ2K
	case "tq3a":
		return ml.DTypeTQ3A
	case "tq3ka":
		return ml.DTypeTQ3KA
	case "tq2a":
		return ml.DTypeTQ2A
	case "tq2ka":
		return ml.DTypeTQ2KA
	case "tq3qa":
		return ml.DTypeTQ3QA
	case "tq2qa":
		return ml.DTypeTQ2QA
	case "tq4":
		return ml.DTypeTQ4
	case "tq4k":
		return ml.DTypeTQ4K
	case "tq4a":
		return ml.DTypeTQ4A
	case "tq4ka":
		return ml.DTypeTQ4KA
	case "tq4qa":
		return ml.DTypeTQ4QA
	case "q8k":
		return ml.DTypeQ8K
	case "q8kv":
		return ml.DTypeQ8KV
	case "q4k":
		return ml.DTypeQ4K
	case "q4kv":
		return ml.DTypeQ4KV
	default:
		return ml.DTypeF16
	}
}

// snapVRAM returns the current GPU memory used in MiB for the given GPU index,
// or -1 if no supported GPU management tool is available (nvidia-smi or rocm-smi).
func snapVRAM(gpuIdx int) int64 {
	// NVIDIA path
	if out, err := exec.Command("nvidia-smi",
		"--query-gpu=memory.used",
		"--format=csv,noheader,nounits",
		fmt.Sprintf("-i=%d", gpuIdx)).Output(); err == nil {
		if v, err := strconv.ParseInt(strings.TrimSpace(string(out)), 10, 64); err == nil {
			return v
		}
	}
	// AMD ROCm path: rocm-smi --showmeminfo vram --csv
	// Output: "device,VRAM Total Memory (B),VRAM Total Used Memory (B)\ncardN,total,used\n..."
	if out, err := exec.Command("rocm-smi", "--showmeminfo", "vram", "--csv").Output(); err == nil {
		lines := strings.Split(strings.TrimSpace(string(out)), "\n")
		for i, line := range lines {
			if i == 0 {
				continue // header
			}
			parts := strings.SplitN(strings.TrimSpace(line), ",", 3)
			if len(parts) != 3 {
				continue
			}
			// card index from "cardN"
			cardStr := strings.TrimPrefix(strings.TrimSpace(parts[0]), "card")
			idx, err := strconv.Atoi(cardStr)
			if err != nil || idx != gpuIdx {
				continue
			}
			usedBytes, err := strconv.ParseInt(strings.TrimSpace(parts[2]), 10, 64)
			if err != nil {
				continue
			}
			return usedBytes / (1024 * 1024)
		}
	}
	return -1
}

// detectGPUIdx picks the GPU with the most free memory, which is the one ggml
// will load the model onto. Tries nvidia-smi then rocm-smi.
func detectGPUIdx() int {
	// NVIDIA path
	if out, err := exec.Command("nvidia-smi",
		"--query-gpu=index,memory.free",
		"--format=csv,noheader,nounits").Output(); err == nil {
		bestIdx := 0
		bestFree := int64(-1)
		for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
			parts := strings.SplitN(strings.TrimSpace(line), ",", 2)
			if len(parts) != 2 {
				continue
			}
			idx, err1 := strconv.Atoi(strings.TrimSpace(parts[0]))
			free, err2 := strconv.ParseInt(strings.TrimSpace(parts[1]), 10, 64)
			if err1 != nil || err2 != nil {
				continue
			}
			if free > bestFree {
				bestFree = free
				bestIdx = idx
			}
		}
		return bestIdx
	}
	// AMD ROCm path
	if out, err := exec.Command("rocm-smi", "--showmeminfo", "vram", "--csv").Output(); err == nil {
		bestIdx := 0
		bestFree := int64(-1)
		lines := strings.Split(strings.TrimSpace(string(out)), "\n")
		for i, line := range lines {
			if i == 0 {
				continue
			}
			parts := strings.SplitN(strings.TrimSpace(line), ",", 3)
			if len(parts) != 3 {
				continue
			}
			cardStr := strings.TrimPrefix(strings.TrimSpace(parts[0]), "card")
			idx, err := strconv.Atoi(cardStr)
			if err != nil {
				continue
			}
			total, err1 := strconv.ParseInt(strings.TrimSpace(parts[1]), 10, 64)
			used, err2 := strconv.ParseInt(strings.TrimSpace(parts[2]), 10, 64)
			if err1 != nil || err2 != nil {
				continue
			}
			free := total - used
			if free > bestFree {
				bestFree = free
				bestIdx = idx
			}
		}
		return bestIdx
	}
	return 0
}

func runForward(m model.Model, inToks []int32, positions []int32, outPositions []int32) []float32 {
	n := len(inToks)
	nOut := len(outPositions)
	ctx := m.Backend().NewContext()
	ctx.SetBatchSize(n)
	batch := input.Batch{
		Positions: positions,
		Sequences: make([]int, n),
	}
	batch.Inputs = ctx.Input().FromInts(inToks, n)
	batch.Outputs = ctx.Input().FromInts(outPositions, nOut)
	logitTensor, err := model.Forward(ctx, m, batch)
	if err != nil {
		log.Fatalf("forward: %v", err)
	}
	ctx.Compute(logitTensor)
	logits := logitTensor.Floats()
	ctx.Close()
	return logits
}

func main() {
	modelPath := flag.String("model", "", "path to GGUF model file (required)")
	preset := flag.String("preset", "f16", "KV cache preset (f16, tq2, tq3, tq4, tq*k, tq*a, tq*ka, tq*qa, ...)")
	ctxLen := flag.Int("ctx", 512, "context window in tokens (KV cache capacity per chunk)")
	maxBatch := flag.Int("max-batch", 512, "physical batch size for prefill (sizes the compute graph; must be ≤ ctx)")
	gpuLayers := flag.Int("gpu-layers", 200, "number of layers to offload to GPU (0 = CPU only)")
	inputFile := flag.String("input", "", "input text file (default: stdin)")
	flashAttn := flag.Bool("flash-attention", true, "enable flash attention (required for TQ presets)")
	decodeSteps := flag.Int("decode-steps", 16, "number of decode steps to time (after warmup)")
	decodeWarmup := flag.Int("decode-warmup", 3, "untimed warmup decode steps before measuring")
	flag.Parse()
	if *maxBatch > *ctxLen {
		*maxBatch = *ctxLen
	}

	if *modelPath == "" {
		log.Fatal("--model is required")
	}

	faType := ml.FlashAttentionEnabled
	if !*flashAttn {
		faType = ml.FlashAttentionDisabled
	}
	if *preset != "f16" && !*flashAttn {
		log.Printf("warning: --flash-attention=false with preset %q: TQ requires FA; result will be f16 PPL", *preset)
	}

	gpuIdx := detectGPUIdx()

	// VRAM baseline — before anything is loaded onto the GPU.
	vramBaseline := snapVRAM(gpuIdx)

	params := ml.BackendParams{
		AllocMemory:    true,
		NumThreads:     runtime.NumCPU(),
		FlashAttention: faType,
	}
	if *gpuLayers > 0 {
		layers := make([]int, *gpuLayers)
		for i := range layers {
			layers[i] = i
		}
		gpus := ggmlbackend.GPUDevices()
		if len(gpus) > 0 {
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
	if err := m.Backend().Load(context.Background(), nil); err != nil {
		log.Fatalf("load weights: %v", err)
	}
	fmt.Fprintln(os.Stderr, "done")

	// VRAM after model weights are on GPU (before KV cache and compute graph).
	vramWeights := snapVRAM(gpuIdx)

	// Set up KV cache.
	// Add decode headroom so single-token decode steps don't overflow capacity.
	decodeHeadroom := *decodeWarmup + *decodeSteps + 4
	cacheCapacity := *ctxLen + decodeHeadroom

	cache := m.Config().Cache
	dtype := ml.DTypeF16
	var tqCache *kvcache.TurboQuantCache
	if cache != nil && *preset != "f16" {
		dt := kvDTypeFromStr(*preset)
		if p, ok := kvcache.PresetFromDType(dt); ok {
			// TQ / q8k / q4k path: wrap with TurboQuantCache.
			wrapped, active := kvcache.WrapWithTurboQuant(cache, p)
			if active {
				cache = wrapped
				if tqc, ok := wrapped.(*kvcache.TurboQuantCache); ok {
					m.SetCache(tqc)
					tqCache = tqc
				}
			}
		} else {
			// Standard quantized KV (q8_0, q4_0): pass dtype to Init directly.
			dtype = dt
		}
	}
	if cache != nil {
		cache.Init(m.Backend(), dtype, 1 /*maxSequences*/, cacheCapacity, *maxBatch)
	}

	tok, ok := m.(tokenizer.Tokenizer)
	if !ok {
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

	tokens, err := tok.Encode(text, false /*addSpecial*/)
	if err != nil {
		log.Fatalf("tokenize: %v", err)
	}
	fmt.Fprintf(os.Stderr, "preset=%s  tokens=%d  ctx=%d  max_batch=%d\n", *preset, len(tokens), *ctxLen, *maxBatch)

	var totalNLL float64
	var totalCount int
	var vramAfterFirstForward int64
	firstForward := true

	// Prefill timing: accumulated across all chunks.
	var prefillTokensTotal int
	var prefillDurTotal time.Duration

	for start := 0; start+1 < len(tokens); start += *ctxLen {
		end := min(start+*ctxLen, len(tokens))
		chunk := tokens[start:end]
		n := len(chunk)
		if n < 2 {
			break
		}

		if cache != nil {
			if err := cache.Remove(0, 0, math.MaxInt32); err != nil {
				log.Fatalf("cache.Remove: %v", err)
			}
		}

		// Chunked prefill: process the chunk in sub-batches of maxBatch tokens.
		// This keeps the compute graph sized for maxBatch regardless of ctx length,
		// matching llama.cpp's n_batch/n_ctx split.
		//
		// NLL is accumulated per-sub-batch so the full logit buffer is never kept
		// in RAM (at vocab=128k a full 32k chunk would require ~17 GiB of RAM).
		// logit at absolute position p predicts chunk[p+1]; we skip the very last
		// position (p = n-1) which has no successor in this chunk.
		t0 := time.Now()
		for sbStart := 0; sbStart < n; sbStart += *maxBatch {
			sbEnd := min(sbStart+*maxBatch, n)
			sbLen := sbEnd - sbStart

			inToks := make([]int32, sbLen)
			positions := make([]int32, sbLen)
			outPos := make([]int32, sbLen)
			for i := range sbLen {
				inToks[i] = int32(chunk[sbStart+i])
				positions[i] = int32(sbStart + i)
				outPos[i] = int32(i)
			}

			sbLogits := runForward(m, inToks, positions, outPos)

			if firstForward {
				vramAfterFirstForward = snapVRAM(gpuIdx)
				firstForward = false
				// Warn if fused path is not active (activateGPUEncode now ran).
				// Slow DequantK path is correct but ~10× slower with +222 MiB VRAM.
				if tqCache != nil && !tqCache.FusedEligible() {
					fmt.Fprintf(os.Stderr, "WARNING: preset=%q fused-FA kernel NOT active for this model's headDim — "+
						"DequantK slow path in use (correct but ~10× slower, +222 MiB VRAM). "+
						"Fused kernel supports headDim in {64, 128}.\n", *preset)
				}
			}

			if len(sbLogits) == 0 {
				continue
			}
			vocabSize := len(sbLogits) / sbLen

			// Compute NLL for each position in this sub-batch that has a successor.
			for i := range sbLen {
				absPos := sbStart + i
				if absPos >= n-1 {
					break // last token in chunk — no successor
				}
				target := int(chunk[absPos+1])
				row := sbLogits[i*vocabSize : (i+1)*vocabSize]
				maxV := row[0]
				for _, v := range row[1:] {
					if v > maxV {
						maxV = v
					}
				}
				var sumExp float64
				for _, v := range row {
					sumExp += math.Exp(float64(v - maxV))
				}
				logP := float64(row[target]-maxV) - math.Log(sumExp)
				totalNLL -= logP
				totalCount++
			}
		}
		elapsed := time.Since(t0)

		prefillTokensTotal += n
		prefillDurTotal += elapsed

		fmt.Fprintf(os.Stderr, "  [%d–%d]  chunk_tokens=%d  prefill=%.0f tok/s  running_ppl=%.3f\n",
			start, end-1, n-1,
			float64(n)/elapsed.Seconds(),
			math.Exp(totalNLL/float64(totalCount)))
	}

	// Decode benchmark.
	// Clear the cache, do a short warmup prefill, then time single-token steps.
	var decodeTPS float64
	if cache != nil && len(tokens) > *decodeWarmup+*decodeSteps+1 {
		// Use the full ctxLen as prefix so decode sees realistic KV bandwidth
		// pressure. 32 tokens gave near-zero KV load, masking real differences.
		if err := cache.Remove(0, 0, math.MaxInt32); err != nil {
			log.Fatalf("cache.Remove: %v", err)
		}

		pfLen := min(*ctxLen, len(tokens)-1)
		for sbStart := 0; sbStart < pfLen; sbStart += *maxBatch {
			sbEnd := min(sbStart+*maxBatch, pfLen)
			sbLen := sbEnd - sbStart
			pfToks := make([]int32, sbLen)
			pfPos := make([]int32, sbLen)
			pfOut := make([]int32, 1)
			pfOut[0] = int32(sbLen - 1) // only need last logit
			for i := range sbLen {
				pfToks[i] = int32(tokens[sbStart+i])
				pfPos[i] = int32(sbStart + i)
			}
			runForward(m, pfToks, pfPos, pfOut)
		}

		// Warmup decode steps (untimed).
		pos := pfLen
		for range *decodeWarmup {
			if pos >= len(tokens) {
				break
			}
			tok := []int32{int32(tokens[pos])}
			p := []int32{int32(pos)}
			runForward(m, tok, p, []int32{0})
			pos++
		}

		// Timed decode steps.
		measuredSteps := min(*decodeSteps, len(tokens)-pos)
		if measuredSteps > 0 {
			t0 := time.Now()
			for range measuredSteps {
				if pos >= len(tokens) {
					break
				}
				tok := []int32{int32(tokens[pos])}
				p := []int32{int32(pos)}
				runForward(m, tok, p, []int32{0})
				pos++
			}
			decodeTPS = float64(measuredSteps) / time.Since(t0).Seconds()
		}
	}

	// Summarise.
	ppl := math.Exp(totalNLL / float64(totalCount))
	prefillTPS := float64(prefillTokensTotal) / prefillDurTotal.Seconds()

	// VRAM deltas.
	totalModelMiB := int64(-1)
	totalActiveMiB := int64(-1)
	if vramBaseline >= 0 && vramWeights >= 0 {
		totalModelMiB = vramWeights - vramBaseline
	}
	if vramBaseline >= 0 && vramAfterFirstForward >= 0 {
		totalActiveMiB = vramAfterFirstForward - vramBaseline
	}

	// True KV cache size: sum Cache[] from each GPU DeviceMemory.
	// This excludes compute-graph scratch (tracked separately in .Graph),
	// so it reflects only the persistent compressed KV buffers.
	kvMiB := int64(0)
	bm := m.Backend().BackendMemory()
	for _, gpu := range bm.GPUs {
		for _, c := range gpu.Cache {
			kvMiB += int64(c)
		}
	}
	kvMiB = (kvMiB + (1 << 19)) >> 20 // round to MiB

	fmt.Printf("preset=%-8s  ppl=%7.4f  eval_tokens=%d  prefill=%6.0f tok/s  decode=%5.1f tok/s  kv_mib=%4d  model_mib=%4d  total_mib=%4d\n",
		*preset, ppl, totalCount,
		prefillTPS, decodeTPS,
		kvMiB, totalModelMiB, totalActiveMiB)
}
