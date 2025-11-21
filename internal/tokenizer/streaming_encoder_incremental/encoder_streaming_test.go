package streaming_encoder_incremental

import (
    "bytes"
    "math/rand"
    "testing"
    "time"

    "github.com/bpetok/internal/tokenizer/core"
)

func encodeStreamingV2(t *testing.T, tok *core.Tokenizer, chunkSizes []int, input []byte) []int {
    enc := NewStreamingEncoderV2(tok)
    out := make([]int, 0, len(input))

    consumed := 0
    for _, size := range chunkSizes {
        if consumed >= len(input) { break }

        end := consumed + size
        if end > len(input) {
            end = len(input)
        }

        chunk := input[consumed:end]
        consumed = end

        toks := enc.Push(chunk)
        if toks != nil {
            out = append(out, toks...)
        }
    }

    // Final flush.
    tail := enc.Flush()
    if tail != nil {
        out = append(out, tail...)
    }

    return out
}

func TestStreamingV2_MatchesOffline(t *testing.T) {
	tok, err := core.LoadTokenizerFromFiles("../testdata/gpt2/vocab.json", "../testdata/gpt2/merges.txt")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}

    input := []byte("hello world! this is a streaming bpe test across chunk boundaries.")

    offline := tok.EncodeOffline(input)

    chunkSizes := []int{1, 2, 3, 5, 8, 13, 32, 64, 128, len(input)}
    got := encodeStreamingV2(t, tok, chunkSizes, input)

    if !bytes.Equal(intSliceToBytes(offline), intSliceToBytes(got)) {
        t.Fatalf("streaming mismatch\noffline: %v\nstreaming: %v", offline, got)
    }
}

func TestStreamingV2_FuzzRandom(t *testing.T) {
	tok, err := core.LoadTokenizerFromFiles("../testdata/gpt2/vocab.json", "../testdata/gpt2/merges.txt")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}

    rand.Seed(time.Now().UnixNano())

    for iter := 0; iter < 200; iter++ {
        n := rand.Intn(200) + 1
        input := make([]byte, n)
        for i := range input {
            input[i] = byte(rand.Intn(256))
        }

        offline := tok.EncodeOffline(input)

        // Random chunk sizes
        var chunks []int
        rem := n
        for rem > 0 {
            sz := rand.Intn(15) + 1
            if sz > rem {
                sz = rem
            }
            chunks = append(chunks, sz)
            rem -= sz
        }

        got := encodeStreamingV2(t, tok, chunks, input)

        if !bytes.Equal(intSliceToBytes(offline), intSliceToBytes(got)) {
            t.Fatalf("mismatch on fuzz input=%v\nchunks=%v\noffline=%v\ngot=%v",
                input, chunks, offline, got)
        }
    }
}

func TestStreamingV2_ChunkBoundaries(t *testing.T) {
	tok, err := core.LoadTokenizerFromFiles("../testdata/gpt2/vocab.json", "../testdata/gpt2/merges.txt")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}

    input := []byte("aaaaaaaabbbbbbbbccccccccddddddddeeeeeeee")

    tests := [][]int{
        {1},                // byte-by-byte
        {2},                // small repeated chunks
        {3}, {4}, {7},
        {8, 8, 8, 8},       // even slices
        {5, 10, 7, 3, 2},   // random
        {32},               // one-shot almost streaming
    }

    offline := tok.EncodeOffline(input)

    for _, chunkSizes := range tests {
        got := encodeStreamingV2(t, tok, chunkSizes, input)
        if !bytes.Equal(intSliceToBytes(offline), intSliceToBytes(got)) {
            t.Fatalf("mismatch at chunkSizes=%v", chunkSizes)
        }
    }
}

func TestStreamingV2_EdgeCases(t *testing.T) {
	tok, err := core.LoadTokenizerFromFiles("../testdata/gpt2/vocab.json", "../testdata/gpt2/merges.txt")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}

    inputs := [][]byte{
        {},
        {0},
        {255},
        []byte("a"),
        []byte("hello"),
        bytes.Repeat([]byte("a"), 1000),
    }

    for _, input := range inputs {
        offline := tok.EncodeOffline(input)
        got := encodeStreamingV2(t, tok, []int{1}, input)

        if !bytes.Equal(intSliceToBytes(offline), intSliceToBytes(got)) {
            t.Fatalf("edge-case mismatch for input=%v", input)
        }
    }
}

func intSliceToBytes(xs []int) []byte {
    b := make([]byte, len(xs)*4)
    for i, v := range xs {
        u := uint32(v)
        b[i*4+0] = byte(u >> 24)
        b[i*4+1] = byte(u >> 16)
        b[i*4+2] = byte(u >> 8)
        b[i*4+3] = byte(u)
    }
    return b
}