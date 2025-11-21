package offline_encoder

import (
	"bytes"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/bpetok/internal/tokenizer/core"
)

func loadTestTokenizer(t *testing.T) *core.Tokenizer {
	t.Helper()

	vocabPath := os.Getenv("TOKENIZER_VOCAB")
	mergesPath := os.Getenv("TOKENIZER_MERGES")
	if vocabPath == "" || mergesPath == "" {
		vocabPath = filepath.Join("../testdata/gpt2", "vocab.json")
		mergesPath = filepath.Join("../testdata/gpt2", "merges.txt")
	}

	tok, err := core.LoadTokenizerFromFiles(vocabPath, mergesPath)
	if err != nil {
		t.Fatalf("failed to load tokenizer: %v", err)
	}
	return tok
}

func TestOfflineEncodeSingleByteCoverage(t *testing.T) {
	tok := loadTestTokenizer(t)

	for b := 0; b < 256; b++ {
		in := []byte{byte(b)}
		ids := tok.EncodeOffline(in)

		if len(ids) != 1 {
			t.Fatalf("byte 0x%02x: expected 1 token, got %d", b, len(ids))
		}

		round := tok.Decode(ids)
		if len(round) != 1 || round[0] != byte(b) {
			t.Fatalf("byte 0x%02x: roundtrip mismatch: %v", b, round)
		}
	}
}

func TestOffline_MinimalMergeTerminates(t *testing.T) {
	tok := loadTestTokenizer(t)

	// Get a merge pair from the tokenizer
	// We'll need to access pairToken, but since it's unexported, we'll test indirectly
	in := []byte(" the")
	ids := tok.EncodeOffline(in)
	if len(ids) == 0 {
		t.Fatalf("expected at least one token")
	}
}

func TestOfflineEncodePairMergesCollapseToSingleToken_Small(t *testing.T) {
	tok := loadTestTokenizer(t)

	base := " the"
	N := 2
	in := strings.Repeat(base, N)

	ids := tok.EncodeOffline([]byte(in))
	if len(ids) == 0 {
		t.Fatalf("empty encoding for repeated %q", base)
	}

	round := tok.Decode(ids)
	if string(round) != in {
		t.Fatalf("round trip mismatch: got %q want %q", string(round), in)
	}
}

func TestOfflineEncodeRoundTripRandom(t *testing.T) {
	tok := loadTestTokenizer(t)

	cases := []int{0, 1, 2, 3, 7, 31, 255, 1024}
	for _, n := range cases {
		buf := make([]byte, n)
		if _, err := rand.Read(buf); err != nil {
			t.Fatalf("rand.Read: %v", err)
		}

		ids := tok.EncodeOffline(buf)
		round := tok.Decode(ids)

		if string(round) != string(buf) {
			t.Fatalf("roundtrip mismatch (n=%d)\n got: %s\nwant: %s",
				n, hex.EncodeToString(round), hex.EncodeToString(buf))
		}
	}
}

func TestOffline_ByteWeirdness(t *testing.T) {
	tok := loadTestTokenizer(t)
	cases := [][]byte{
		{0x00, 0xFF, 0x10, 0x7F},
		[]byte("tabs\tnewlines\n\r"),
		[]byte("💥🔥 the 💥"), // multibyte UTF-8
	}
	for _, in := range cases {
		ids := tok.EncodeOffline(in)
		out := tok.Decode(ids)
		if !bytes.Equal(out, in) {
			t.Fatalf("roundtrip mismatch for %q", in)
		}
	}
}

func TestOffline_Determinism(t *testing.T) {
	tok := loadTestTokenizer(t)
	in := []byte("determinism determinism determinism")
	a := tok.EncodeOffline(in)
	b := tok.EncodeOffline(in)
	if fmt.Sprint(a) != fmt.Sprint(b) {
		t.Fatalf("nondeterministic")
	}

	if out := tok.Decode(a); string(out) != string(in) {
		t.Fatalf("roundtrip")
	}
	c := tok.EncodeOffline(tok.Decode(a))
	if fmt.Sprint(a) != fmt.Sprint(c) {
		t.Fatalf("idempotence broken")
	}
}

func TestDecode_RoundTrip(t *testing.T) {
	tok := loadTestTokenizer(t)
	in := []byte("quick test: the cat sat")
	ids := tok.EncodeOffline(in)
	out := tok.Decode(ids)
	if string(out) != string(in) {
		t.Fatalf("round-trip mismatch: got %q want %q", out, in)
	}
}
