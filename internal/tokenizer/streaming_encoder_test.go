package tokenizer

import (
	"math/rand"
	"testing"
	"time"
)

func encodeStreamingNaive(t *testing.T, tok *Tokenizer, input []byte, chunkSizes []int) []int {
	t.Helper()
	es := NewEncoderState(tok)
	var out []int

	if len(chunkSizes) == 0 {
		chunkSizes = []int{len(input)}
	}

	pos := 0
	idx := 0
	lastIdx := len(chunkSizes) - 1
	for pos < len(input) {
		sz := chunkSizes[idx]
		if sz <= 0 {
			break
		}

		if idx < lastIdx {
			idx++
		}

		end := pos + sz
		if end > len(input) {
			end = len(input)
		}

		emitted := es.Push(input[pos:end])
		out = append(out, emitted...)
		pos = end

		if idx >= len(chunkSizes) {
			idx = lastIdx
		}
	}

	out = append(out, es.Flush()...)
	return out
}

func TestNaiveStreamingMatchesGreedy_SimpleChunkings(t *testing.T) {
	tok := loadTestTokenizer(t)

	cases := []struct {
		name string
		s    string
	}{
		{"empty", ""},
		{"ascii_short", "hello world"},
		{"ascii_punct", "hello, world! this is bpe-tok :)"},
		{"utf8_simple", "Hello 你好 नमस्ते"},
		{"emoji", "hi 👋🏽 this is  tokenizer"},
		{"repeated_patterns", "aaaaaaabaaaaaaabaaaaaaab"},
	}

	chunkings := [][]int{
		{len("hello world")},     // 1 chunk (whole string)
		{1},                      // 1 byte at a time
		{2},                      // 2 bytes at a time
		{3},                      // 3 bytes at a time
		{4, 4, 4, 4, 4, 4, 4, 4}, // fixed window, overrun is okay
	}

	for _, tc := range cases {
		for i, chunks := range chunkings {
			input := []byte(tc.s)

			want := tok.EncodeOffline(input)
			got := encodeStreamingNaive(t, tok, input, chunks)

			if !equalIntSlices(want, got) {
				t.Fatalf("case %q chunking %d: mismatch.\nwant: %v\ngot:  %v",
					tc.name, i, want, got)
			}
		}
	}
}

func TestStreamingMatchesGreedy_Randomized(t *testing.T) {
	tok := loadTestTokenizer(t)

	const (
		numCases  = 200
		maxLen    = 256
		maxChunks = 16
	)
	r := rand.New(rand.NewSource(time.Now().UnixNano()))

	for caseIdx := 0; caseIdx < numCases; caseIdx++ {
		// random input
		n := r.Intn(maxLen + 1)
		input := make([]byte, n)
		for i := 0; i < n; i++ {
			if r.Float64() < 0.8 {
				input[i] = byte(32 + r.Intn(95))
			} else {
				input[i] = byte(r.Intn(256))
			}
		}

		var chunkSizes []int
		if n == 0 {
			chunkSizes = []int{0}
		} else {
			remaining := n
			for remaining > 0 && len(chunkSizes) < maxChunks {
				sz := 1 + r.Intn(remaining)
				chunkSizes = append(chunkSizes, sz)
				remaining -= sz
			}
			if r.Float64() < 0.3 {
				chunkSizes = append(chunkSizes, 1+r.Intn(maxLen))
			}
		}

		want := tok.EncodeOffline(input)
		got := encodeStreamingNaive(t, tok, input, chunkSizes)

		if !equalIntSlices(want, got) {
			t.Fatalf("random case %d: mismatch\ninput: %q\nchunks: %v\nwant: %v\ngot: %v",
				caseIdx, string(input), chunkSizes, want, got)
		}
	}
}

func equalIntSlices(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}

	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}

	return true
}
