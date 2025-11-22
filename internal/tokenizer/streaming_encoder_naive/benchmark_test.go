package streaming_encoder_naive

import (
	"os"
	"sync"
	"testing"

	"github.com/bpetok/internal/tokenizer/core"
)

func mustLoadBenchCorpus(b *testing.B, path string) []byte {
	b.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		b.Fatalf("failed to read test data %q: %v", path, err)
	}
	return data
}

func BenchmarkNaiveEncodeStreaming_8Parallel_4KBChunks(b *testing.B) {
	tok := loadTestTokenizerB(b)
	input := mustLoadBenchCorpus(b, "../testdata/gpt2/bench_corpus.txt")

	const chunkSize = 4 << 10         // 4 KiB
	b.SetBytes(int64(len(input)) * 8) // total bytes processed across 8 streams

	b.ResetTimer()

	for n := 0; n < b.N; n++ {
		var wg sync.WaitGroup
		wg.Add(8)

		for streamID := 0; streamID < 8; streamID++ {
			go func() {
				defer wg.Done()

				es := NewNaiveStreamingEncoderState(tok)

				pos := 0
				for pos < len(input) {
					end := pos + chunkSize
					if end > len(input) {
						end = len(input)
					}
					_ = es.Push(input[pos:end])
					pos = end
				}

				_ = es.Flush()
			}()
		}

		wg.Wait()
	}
}

func BenchmarkNaiveEncodeStreaming_WholeChunk(b *testing.B) {
	tok := loadTestTokenizerB(b)
	input := mustLoadBenchCorpus(b, "../testdata/gpt2/bench_corpus.txt")

	b.SetBytes(int64(len(input)))
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		es := NewNaiveStreamingEncoderState(tok)
		_ = es.Push(input)
		_ = es.Flush()
	}
}

func BenchmarkNaiveEncodeStreaming_4KBChunks(b *testing.B) {
	tok := loadTestTokenizerB(b)
	input := mustLoadBenchCorpus(b, "../testdata/gpt2/bench_corpus.txt")

	const chunkSize = 4 << 10 // 4 KiB
	b.SetBytes(int64(len(input)))
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		es := NewNaiveStreamingEncoderState(tok)
		var pos int
		for pos < len(input) {
			end := pos + chunkSize
			if end > len(input) {
				end = len(input)
			}
			_ = es.Push(input[pos:end])
			pos = end
		}
		_ = es.Flush()
	}
}

func loadTestTokenizerB(b *testing.B) *core.Tokenizer {
	b.Helper()
	tok, err := core.LoadTokenizerFromFiles(
		"../testdata/gpt2/vocab.json",
		"../testdata/gpt2/merges.txt",
	)
	if err != nil {
		b.Fatalf("failed to load tokenizer: %v", err)
	}
	return tok
}
