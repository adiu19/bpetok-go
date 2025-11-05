package tokenizer

import (
	"bufio"
	"container/heap"
	"encoding/json"
	"fmt"
	"os"
	"strings"
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
	pairRank map[[2]int]int
	// given two adjacent tokens (A,B), what's the merged token C
	pairToken map[[2]int]int
	// TODO: for streaming
	maxMergeDepth int
}

// LoadTokenizerFromFiles builds a tokenizer from vocab and merges
// vocabPath and mergesPath are raw file paths
func LoadTokenizerFromFiles(vocabPath, mergesPath string) (*Tokenizer, error) {
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

	revVocab, err := buildRevVocab(vocab, len(vocab))
	if err != nil {
		return nil, fmt.Errorf("failed to build revVocab: %w", err)
	}

	byteToToken, err := buildByteToToken(revVocab)
	if err != nil {
		return nil, fmt.Errorf("failed to build bytesToToken : %w", err)
	}

	// --------------------------------------------------------------------
	// ---------------------------------------------------- onto merges now
	// --------------------------------------------------------------------

	mergesLines, err := readLines(mergesPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read mergs: %w", err)
	}

	pairRank, err := buildPairRank(mergesLines, vocab)
	if err != nil {
		return nil, fmt.Errorf("error while building pairRank : %w", err)
	}

	pairToken, err := buildPairToken(revVocab, pairRank)
	if err != nil {
		return nil, fmt.Errorf("failed to build pairToken : %w", err)
	}

	return &Tokenizer{
		revVocab:      revVocab,
		byteToToken:   byteToToken,
		pairRank:      pairRank,
		pairToken:     pairToken,
		maxMergeDepth: 0,
	}, nil

}

// buildByteToToken constructs the [256]int lookup table that maps a single raw
// byte value (0..255) to the token ID that represents exactly that byte.
func buildByteToToken(revVocab [][]byte) ([256]int, error) {
	var table [256]int

	filled := [256]bool{}
	for tokenID, bs := range revVocab {
		if len(bs) == 1 {
			b := bs[0]

			if filled[b] {
				return table, fmt.Errorf("duplicate single byte token")
			}

			table[b] = tokenID
			filled[b] = true
		}

	}

	for b := 0; b < 256; b++ {
		if !filled[b] {
			return table, fmt.Errorf("no token found for raw bytes %d", int(b))
		}
	}

	return table, nil
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
//	For each rune r:
//	if string(r) is in byteDecoder: append that decoded byte
//	else: append the UTF-8 encoding of r directly
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

// buildPairRank assigns a rank (0 being highest) to each pair of tokens in the merges dataset
// the merges dataset comes to us as a pair of utf-8 encoded strings, which we map to token ids using vocab
// the function also contains a validation step that ensures merges doesn't contain duplicate entries
func buildPairRank(mergesLines []string, vocabMap map[string]int) (map[[2]int]int, error) {
	pairRank := make(map[[2]int]int, len(mergesLines))

	rank := 0
	for _, line := range mergesLines {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") { // skip this line
			continue
		}
		parts := strings.Fields(line)
		if len(parts) != 2 {
			return nil, fmt.Errorf("invalid merge line %q, we want exactly two items per line", line)
		}

		leftStr := parts[0]
		rightStr := parts[1]

		leftID, ok1 := vocabMap[leftStr]
		rightID, ok2 := vocabMap[rightStr]

		if !ok1 || !ok2 {
			return nil, fmt.Errorf("failed to find a vocab entry for an entry in merges. left: %v, right: %v", &leftStr, &rightStr)
		}

		key := [2]int{leftID, rightID}
		if _, exists := pairRank[key]; exists {
			return nil, fmt.Errorf("duplicate merge pair found in given merges, %v", key)
		}

		pairRank[key] = rank
		rank++
	}

	return pairRank, nil
}

// buildPairToken builds a mapping structure that maps a pair of token ids proposed by merges rules to an output token id
func buildPairToken(revVocab [][]byte, pairRank map[[2]int]int) (map[[2]int]int, error) {
	// init bytesToID, which is our temporary "reverse mapping of revVocab"
	bytesToID := make(map[string]int, len(revVocab))
	for id, bs := range revVocab {
		bytesToID[string(bs)] = id
	}

	pairToken := make(map[[2]int]int, len(pairRank))

	for pair := range pairRank {
		leftID := pair[0]
		rightID := pair[1]

		if leftID < 0 || leftID >= len(revVocab) ||
			rightID < 0 || rightID >= len(revVocab) {
			return nil, fmt.Errorf("the token id parsed from pair ranks is out of bounds leftID: %d, rightID: %d", leftID, rightID)
		}

		leftBytes := revVocab[leftID]
		rightBytes := revVocab[rightID]

		mergedBytes := make([]byte, 0, len(leftBytes)+len(rightBytes))
		mergedBytes = append(mergedBytes, leftBytes...)
		mergedBytes = append(mergedBytes, rightBytes...)

		mergedID, ok := bytesToID[string(mergedBytes)]
		if !ok {
			return nil, fmt.Errorf("error mapping concatenated bytes to a valid token id based off of revVocab %s", string(mergedBytes))

		}

		if _, exists := pairToken[pair]; exists {
			return nil, fmt.Errorf("duplicate pair in pairToken for %v", pair)
		}

		pairToken[pair] = mergedID

	}

	return pairToken, nil
}

// readLines reads a text file into []string whilst preserving order
func readLines(path string) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var out []string
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		out = append(out, sc.Text())
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return out, nil
}

// EncodeOffline takes a sequence of bytes and converts them to sequence of tokens
func (t *Tokenizer) EncodeOffline(input []byte) []int {
	n := len(input)
	if n == 0 {
		return nil
	}

	tokens := make([]int, n)

	// convert the input to tokens, where each token currently represents a single byte
	for i, b := range input {
		tokens[i] = t.byteToToken[b]
	}

	// doubly linked-list
	prev := make([]int, n)
	next := make([]int, n)
	for i := 1; i < n-1; i++ {
		prev[i] = i - 1
		next[i] = i + 1
	}

	// edge elements
	prev[0] = -1
	next[n-1] = -1

	// per-slot versioning to invalidate heap entries
	liveVersion := make([]int, n)

	h := &mergeHeap{}
	heap.Init(h)

	// seed heap with all initial adjacent pairs.
	pushIfMergeable := func(i int) {
		j := next[i]
		if i == -1 || j == -1 {
			// not a valid index
			return
		}

		a := tokens[i]
		b := tokens[j]

		if rank, ok := t.pairRank[[2]int{a, b}]; ok {
			heap.Push(h, mergeCand{
				rank:       rank,
				pos:        i,
				leftToken:  a,
				rightToken: b,
				verL:       liveVersion[i],
				verR:       liveVersion[j],
			})
		}
	}

	// loop that fills heap with initial seed
	for i := 0; i != -1 && next[i] != -1; i = next[i] {
		pushIfMergeable(i)
	}

	// leftmost index (never dies; we always merge into the left slot)
	head := 0

	// while there are still entries in our heap
	for h.Len() > 0 {
		c := heap.Pop(h).(mergeCand)
		i := c.pos
		if i == -1 {
			continue
		}

		j := next[i]
		if j == -1 {
			continue // no right neibhbor anymore
		}

		// stale entry since atleast one version did not match
		if liveVersion[i] != c.verL || liveVersion[j] != c.verR {
			continue
		}

		a := tokens[i]
		b := tokens[j]

		rankNow, ok := t.pairRank[[2]int{a, b}]

		// if this entry doesn’t describe the same (a,b) pair with the same rank that it did when it was pushed — skip it
		if !ok || rankNow != c.rank || a != c.leftToken || b != c.rightToken {
			continue
		}

		cID := t.pairToken[[2]int{a, b}]
		tokens[i] = cID // collapse into slot i

		nj := next[j]
		next[i] = nj
		if nj != -1 {
			prev[nj] = i
		}

		// mark other pointers as dead
		prev[j], next[j] = -1, -1

		liveVersion[i]++
		liveVersion[j]++ // j died; invalidate anything mentioning it

		// push this newly created token as part of a new pair back into the heap, with the previous as the first element
		if pi := prev[i]; pi != -1 {
			pushIfMergeable(pi)
		}

		// push this newly created token as part of a new pair back into the heap, with the new token as the first element
		pushIfMergeable(i)
	}

	out := make([]int, 0, n)
	for i := head; i != -1; i = next[i] {
		out = append(out, tokens[i])
	}

	return out
}

// ------------------------------ heap structures
type mergeCand struct {
	rank       int // lower wins
	pos        int // left index; lower wins on tie to enforce leftmost
	leftToken  int
	rightToken int
	verL       int
	verR       int
}

type mergeHeap []mergeCand

func (h mergeHeap) Len() int { return len(h) }
func (h mergeHeap) Less(i, j int) bool {
	if h[i].rank != h[j].rank {
		return h[i].rank < h[j].rank
	}
	return h[i].pos < h[j].pos // leftmost tie-break
}
func (h mergeHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }
func (h *mergeHeap) Push(x any)   { *h = append(*h, x.(mergeCand)) }
func (h *mergeHeap) Pop() any     { old := *h; n := len(old); x := old[n-1]; *h = old[:n-1]; return x }
