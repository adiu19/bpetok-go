package tokenizer

import (
	"encoding/json"
	"fmt"
	"os"
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

// LoadTokenizer builds a tokenizer from vocab and merges
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

// // NewEncoder
// func (t *Tokenizer) NewEncoder() Encoder

// // NewDocoder
// func (t *Tokenizer) NewDocoder() Decoder
