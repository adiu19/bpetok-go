package offline_encoder

import (
	"os"
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

func BenchmarkEncodeOffline(b *testing.B) {
	tok := loadTestTokenizerB(b)
	input := mustLoadBenchCorpus(b, "../testdata/gpt2/bench_corpus.txt")

	b.SetBytes(int64(len(input)))
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = tok.EncodeOffline(input)
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

