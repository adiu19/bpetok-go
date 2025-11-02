package tokenizer

import (
	"encoding/json"
	"fmt"
	"os"
	"unicode/utf8"
)

// Encoder interface
type Encoder interface {
	/*
		Feed consumes the next chunk of raw bytes from the input stream. It may emit zero or more
		completed token IDs.
		The returned slice is allowed to alias internal memory (zero-copy) so the caller must treat it as read-only
		and make a copy if they want edits.
	*/
	Feed(chunk []byte) []int

	/*
		Flush tells the encoder that the stream is complete. It returns any remaining token IDs that were buffered
		because they were being waited on to see if there are more merges to apply on them. After flush, the encoder
		is reset to a clean state and can be reused for a new stream.
	*/
	Flush() []int
}

// Decoder interface, no need for flush right now because we won't be maintaining internal buffer
type Decoder interface {
	/*
		Feed consumes token IDs and returns zero or more decoded bytes. Same as the encoder, there is a zero-copy rule;
		returned slice can alias internal memory, call must treat it as read-only
	*/
	Feed(tokens []int) []byte
}

type bpePair struct {
	A int32
	B int32
}

// Tokenizer holds immutable model data derived from a BPE vocab/merges set which is safe for concurrent use.
// Invariants we maintain:
//   - revVocab[id] is the exact byte sequence for token ID 'id'.
//   - For every byte b in [0..255], byteToToken[b] gives a valid base token ID.
//   - If pairRank[[2]int{A,B}] exists, then pairToken[[2]int{A,B}] exists and
//     pairToken[[2]int{A,B}] = C is the token ID produced by merging A then B.
type Tokenizer struct {
	// for decoding, index = token_id, value is byte sequence
	revVocab [][]byte
	//  seed the first pass of encoder from raw bytes
	//  in a byte-level BPE tokenizer, every possible byte 0..255 must have a mapping.
	byteToToken [256]int
	// pairRank[bpePair{A,B}] is the priority rank for merging token A followed by token B.
	// bpePair value can be used as a map key because fixed-size arrays are comparable
	pairRank map[bpePair]int
	// given two adjacent tokens (A,B), what's the merged token C
	pairToken map[bpePair]int
}

// LoadTokenizerFromFiles builds a tokenizer from vocab and merges
// vocabPath and mergesPath are raw file paths
func LoadTokenizerFromFiles(vocabPath, mergesPath string) (*Tokenizer, error) {
	/*
		step 1: parse vocabBytes into
			- revVocab
			- byteToToken

		step 2: parse mergeBytes into
			- pairRank
			- pairToken

		step 3: return tokenizer ref
	*/
	data, err := os.ReadFile(vocabPath)
	if err != nil {
		return nil, fmt.Errorf("error while reading vocab file : %w", err)
	}

	var vocab map[string]int
	if err := json.Unmarshal(data, &vocab); err != nil {
		return nil, fmt.Errorf("error while unmarshalling vocab: %w", err)
	}

	maxID := -1
	seen := make(map[int]bool)
	for _, id := range vocab {
		seen[id] = true
		if id > maxID {
			maxID = id
		}
	}

	for i := 0; i <= maxID; i++ {
		if !seen[i] {
			return nil, fmt.Errorf("vocab not dense and missing %d", i)
		}
	}
	fmt.Printf("vocabs loaded, %d tokens (0..%d)\n", len(vocab), maxID)

	return &Tokenizer{}, nil

}

// buildRevVocab takes the parsed vocab.json (tokenString -> id) and returns revVocab[id] = raw bytes for that token id.
// vocabSize should be the expected number of IDs (e.g. 50257).
func buildRevVocab(vocab map[string]int, vocabSize int) ([][]byte, error) {
	if len(vocab) != vocabSize {
		return nil, fmt.Errorf("vocab length mismatch. expected %d, received. %d", vocabSize, len(vocab))
	}

	byteDecoder := buildCursedByteDecoder()

	revVocab := make([][]byte, vocabSize)
	for tokenStr, id := range vocab {
		if id < 0 || id >= vocabSize {
			return nil, fmt.Errorf("token id out of range : %d", id)
		}

		tokenBytes, err := decodeTokenString(tokenStr, byteDecoder)
		if err != nil {
			return nil, fmt.Errorf("failed to decode token %s at index %d", tokenStr, id)
		}

		if len(tokenBytes) == 0 {
			return nil, fmt.Errorf("decoded empty byte sequence for token id %d and token string %s", id, tokenStr)
		}

		bcopy := make([]byte, len(tokenBytes)) // allocate a brand-new slice of bytes, initially full of zeroes
		copy(bcopy, tokenBytes)                // copy the actual bytes from tokenBytes into bcopy
		revVocab[id] = bcopy                   // store the copy in revVocab
	}

	// validate all slots
	for i := 0; i < vocabSize; i++ {
		if len(revVocab[i]) == 0 {
			return nil, fmt.Errorf("revVocab[%d] is unset ", i)
		}
	}

	seen := make(map[string]int, vocabSize)
	for id, b := range revVocab {
		k := string(b)
		if prev, exists := seen[k]; exists {
			return nil, fmt.Errorf("duplicate byte sequence found. check id %d and %d", prev, id)
		}
		seen[k] = id
	}
	return revVocab, nil

}

// decodeTokenString turns a vocab.json key (which might contain those weird
// extended unicode stand-ins for bytes) back into the real raw bytes that
// token represents
//
// Rules (GPT-2 style):
//   - During training/export, each raw byte 0..255 got mapped to some "printable-ish"
//     rune sequence so it can live in JSON. That's byteEncoder.
//   - We invert that mapping into byteDecoder: string(rune) -> original byte value.
//   - To recover original bytes for a token string, we walk its runes.
//     For each rune r:
//     if string(r) is in byteDecoder: append that decoded byte
//     else: append the UTF-8 encoding of r directly
//
// Why the "else" branch?
//   - Multi-byte tokens like "the" are stored literally as 't','h','e' etc,
//     so for normal ASCII we can just use their UTF-8 bytes.
//   - But for things like "Ġ", that rune is actually standing in for 0x20 (space).
//
// Result is the real byte sequence that this token ID should output during decode.
func decodeTokenString(s string, byteDecoder map[rune]byte) ([]byte, error) {
	var out []byte

	for len(s) > 0 {
		r, size := utf8.DecodeRuneInString(s)
		if r == utf8.RuneError && size == 1 {
			return nil, fmt.Errorf("invalid utf8 in token string at %q", s)
		}

		if b, ok := byteDecoder[r]; ok {
			// this rune is actually one raw byte
			out = append(out, b)
		} else {
			// otherwise this rune is meant literally, append its UTF-8 bytes
			var tmp [utf8.UTFMax]byte
			n := utf8.EncodeRune(tmp[:], r) // turns that Unicode code point into its UTF-8 byte sequence
			out = append(out, tmp[:n]...)
		}

		s = s[size:] // advance by size
	}

	return out, nil
}

// buildCursedByteDecoder exists because GPT-2 decided JSON was a good idea for serializing 256 arbitrary bytes.
// This function painstakingly replays their “byte → fake Unicode rune” ritual so we can un-serialize vocab.json
// without breaking compatibility.
//
// I learned nothing from this. It is technical debt distilled into code.
// If anyone's reading this, I'm truly sorry.
func buildCursedByteDecoder() map[rune]byte {
	var bs []int
	for b := 33; b <= 126; b++ {
		bs = append(bs, b)
	}
	for b := 161; b <= 172; b++ {
		bs = append(bs, b)
	}
	for b := 174; b <= 255; b++ {
		bs = append(bs, b)
	}

	cs := make([]int, len(bs))
	copy(cs, bs)

	// assign stand-ins (256, 257, ...) for the remaining bytes
	next := 256
	for b := 0; b < 256; b++ {
		found := false
		for _, x := range bs {
			if x == b {
				found = true
				break
			}
		}
		if !found {
			bs = append(bs, b)
			cs = append(cs, next)
			next++
		}
	}

	// build decoder: rune -> byte
	byteDecoder := make(map[rune]byte, 256)
	for i := range bs {
		origByte := byte(bs[i])
		r := rune(cs[i])
		byteDecoder[r] = origByte
	}

	return byteDecoder
}

// // NewEncoder
// func (t *Tokenizer) NewEncoder() Encoder

// // NewDocoder
// func (t *Tokenizer) NewDocoder() Decoder
