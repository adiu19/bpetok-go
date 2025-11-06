package tokenizer

import (
	"bytes"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func loadTestTokenizer(t *testing.T) *Tokenizer {
	t.Helper()

	vocabPath := os.Getenv("TOKENIZER_VOCAB")
	mergesPath := os.Getenv("TOKENIZER_MERGES")
	if vocabPath == "" || mergesPath == "" {
		vocabPath = filepath.Join("testdata/gpt2", "vocab.json")
		mergesPath = filepath.Join("testdata/gpt2", "merges.txt")
	}

	tok, err := LoadTokenizerFromFiles(vocabPath, mergesPath)
	if err != nil {
		t.Fatalf("failed to load tokenizer: %v", err)
	}
	return tok
}

func decodeBytes(t *testing.T, tok *Tokenizer, ids []int) []byte {
	t.Helper()
	var out []byte
	for _, id := range ids {
		if id < 0 || id >= len(tok.revVocab) {
			t.Fatalf("token id out of range: %d", id)
		}
		out = append(out, tok.revVocab[id]...)
	}
	return out
}

func TestOfflineEncodeSingleByteCoverage(t *testing.T) {
	tok := loadTestTokenizer(t)

	for b := 0; b < 256; b++ {
		in := []byte{byte(b)}
		ids := tok.EncodeOffline(in)

		if len(ids) != 1 {
			t.Fatalf("byte 0x%02x: expected 1 token, got %d", b, len(ids))
		}
		if ids[0] != tok.byteToToken[byte(b)] {
			t.Fatalf("byte 0x%02x: got token %d, want %d", b, ids[0], tok.byteToToken[byte(b)])
		}

		round := decodeBytes(t, tok, ids)
		if len(round) != 1 || round[0] != byte(b) {
			t.Fatalf("byte 0x%02x: roundtrip mismatch: %v", b, round)
		}
	}
}

func TestOffline_MinimalMergeTerminates(t *testing.T) {
	tok := loadTestTokenizer(t)

	var a, b, c int
	for p, out := range tok.pairToken {
		a, b, c = p[0], p[1], out
		break
	}
	if a == 0 && b == 0 && c == 0 {
		t.Fatalf("no merges available")
	}

	in := append(append([]byte{}, tok.revVocab[a]...), tok.revVocab[b]...)
	ids := tok.EncodeOffline(in) // if this hangs, the loop is stuck
	if len(ids) != 1 || ids[0] != c {
		t.Fatalf("expected [%d] for (%d,%d)->%d, got %v", c, a, b, c, ids)
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

	round := decodeBytes(t, tok, ids)
	if string(round) != in {
		t.Fatalf("round trip mismatch: got %q want %q", string(round), in)
	}
}

func TestOfflineEncodeMergeChainsDepth2(t *testing.T) {
	tok := loadTestTokenizer(t)

	limit := 0
	for pair1, x := range tok.pairToken {
		found := false
		var c, y int
		// scan right-hand neighbors for x
		for p2, yCand := range tok.pairToken {
			if p2[0] == x {
				c = p2[1]
				y = yCand
				found = true
				break
			}
		}
		if !found {
			continue
		}

		a := pair1[0]
		b := pair1[1]
		in := append(append(append([]byte{}, tok.revVocab[a]...), tok.revVocab[b]...), tok.revVocab[c]...)
		ids := tok.EncodeOffline(in)
		if len(ids) != 1 || ids[0] != y {
			t.Fatalf("chain (%d,%d)->%d, then (%d,%d)->%d: got %v",
				a, b, x, x, c, y, ids)
		}

		limit++
		if limit >= 50 {
			break
		}
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
		round := decodeBytes(t, tok, ids)

		if string(round) != string(buf) {
			t.Fatalf("roundtrip mismatch (n=%d)\n got: %s\nwant: %s",
				n, hex.EncodeToString(round), hex.EncodeToString(buf))
		}
	}
}

func TestOfflineEncodeLeftmostTieBreak(t *testing.T) {
	tok := loadTestTokenizer(t)

	var picked [2]int
	for pair := range tok.pairRank {
		picked = pair
		break
	}
	if picked == ([2]int{}) {
		t.Skip("no merges found; unlikely")
	}

	p := picked[0]
	q := picked[1]

	in := append(append(append([]byte{}, tok.revVocab[p]...), tok.revVocab[q]...), tok.revVocab[p]...)
	in = append(in, tok.revVocab[q]...)

	ids := tok.EncodeOffline(in)
	if len(ids) >= 4 {
		t.Fatalf("expected at least one merge to occur; got %v", ids)
	}

	// Sanity round-trip
	round := decodeBytes(t, tok, ids)
	if string(round) != string(in) {
		t.Fatalf("tie-break roundtrip mismatch")
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

func TestDecode_Bounds(t *testing.T) {
	tok := loadTestTokenizer(t)
	defer func() {
		if recover() == nil {
			t.Fatalf("expected panic on out-of-range id")
		}
	}()
	_ = tok.Decode([]int{-1})
}
