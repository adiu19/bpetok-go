package streaming_encoder_incremental

import (
	"github.com/bpetok/internal/tokenizer/core"
)

type heapInterface interface {
	Push(c mergeCandidate)
	Pop() (mergeCandidate, bool)
	Empty() bool
	Reset()
}

type StreamingEncoderV2 struct {
	tok *core.Tokenizer

	tokens  []int
	prev    []int
	next    []int
	live    []uint32
	liveGen uint32

	heap heapInterface

	head int
	tail int

	outBuf           []int
	tailReserve      int
	syntheticLengths map[int]int
}

func NewStreamingEncoderV2(tok *core.Tokenizer) *StreamingEncoderV2 {
	tok.UseUnicodeInitTokens = true
	maxRank := tok.GetMaxRank()
	return &StreamingEncoderV2{
		tok:         tok,
		head:        -1,
		tail:        -1,
		liveGen:     1,
		outBuf:      make([]int, 0, 128),
		heap:        newMergeHeapWithMaxRank(maxRank),
		tailReserve: tok.MaxTokenByteLen - 1,
	}
}

func (se *StreamingEncoderV2) Push(chunk []byte) []int {
	if len(chunk) == 0 {
		return nil
	}

	se.heap.Reset()

	oldTail := se.tail

	newNodes := se.appendBytes(chunk)
	if len(newNodes) == 0 {
		return nil
	}

	if oldTail != -1 {
		se.maybeAddCandidate(oldTail, newNodes[0])
	}

	se.seedAdjacency(newNodes)

	se.runMerges()

	out := []int{}

	se.commitStablePrefix(&out)

	if se.tail != -1 {
		firstLive := -1
		for _, idx := range newNodes {
			if se.live[idx] != 0 {
				firstLive = idx
				break
			}
		}
		if firstLive != -1 {
			se.maybeAddCandidate(se.tail, firstLive)
		}
	}

	if len(out) == 0 {
		return nil
	}
	return out
}

func (se *StreamingEncoderV2) Flush() []int {
	if se.head == -1 {
		return nil
	}

	out := make([]int, 0, 16)
	se.commitPrefix(&out)

	buf := make([]byte, 0, 64)
	for idx := se.head; idx != -1; idx = se.next[idx] {
		tokID := se.tokens[idx]
		buf = append(buf, se.tok.RevVocab[tokID]...)
	}

	if len(buf) > 0 {
		rem := se.tok.EncodeOffline(buf, nil)
		out = append(out, rem...)
	}

	se.head = -1
	se.tail = -1

	se.heap = newMergeHeap()

	return out
}

func (se *StreamingEncoderV2) appendBytes(chunk []byte) []int {
	if len(chunk) == 0 {
		return nil
	}

	count := len(chunk)
	start := len(se.tokens)
	end := start + count - 1

	newIndices := make([]int, count)

	se.tokens = append(se.tokens, make([]int, count)...)
	se.prev = append(se.prev, make([]int, count)...)
	se.next = append(se.next, make([]int, count)...)
	se.live = append(se.live, make([]uint32, count)...)

	for i := 0; i < count; i++ {
		idx := start + i
		newIndices[i] = idx

		se.tokens[idx] = se.tok.GetByteToInitialToken(chunk[i])

		se.liveGen++
		se.live[idx] = se.liveGen
	}

	if se.tail == -1 {
		for i := 0; i < count; i++ {
			idx := start + i
			if i == 0 {
				se.prev[idx] = -1
			} else {
				se.prev[idx] = idx - 1
			}
			if i == count-1 {
				se.next[idx] = -1
			} else {
				se.next[idx] = idx + 1
			}
		}
		se.head = start
		se.tail = end
		return newIndices
	}

	first := start
	se.next[se.tail] = first
	se.prev[first] = se.tail

	for i := 0; i < count; i++ {
		idx := start + i
		if i == count-1 {
			se.next[idx] = -1
			if i > 0 {
				se.prev[idx] = idx - 1
			}
		} else {
			if i > 0 {
				se.prev[idx] = idx - 1
			}
			se.next[idx] = idx + 1
		}
	}

	se.tail = end

	return newIndices
}

func ifElse(cond bool, a, b int) int {
	if cond {
		return a
	}
	return b
}

func (se *StreamingEncoderV2) seedAdjacency(newNodes []int) {
	if len(newNodes) == 0 {
		return
	}

	first := newNodes[0]
	if se.head != first {
		left := se.prev[first]
		if left != -1 {
			se.maybeAddCandidate(left, first)
		}
	}

	for i := 0; i+1 < len(newNodes); i++ {
		left := newNodes[i]
		right := newNodes[i+1]
		se.maybeAddCandidate(left, right)
	}
}

func (se *StreamingEncoderV2) maybeAddCandidate(i, j int) {
	if i == -1 || j == -1 {
		return
	}

	leftTok := se.tokens[i]
	rightTok := se.tokens[j]

	rank, ok := se.tok.GetPairRank(leftTok, rightTok)
	if !ok {
		return
	}

	se.heap.Push(mergeCandidate{
		leftIndex:  i,
		rightIndex: j,
		rank:       rank,
		liveLeft:   se.live[i],
		liveRight:  se.live[j],
	})
}

func (se *StreamingEncoderV2) runMerges() {
	for {
		cand, ok := se.heap.Pop()
		if !ok {
			return
		}

		if !se.isValidCandidate(cand) {
			continue
		}

		se.performMerge(cand)

		se.updateFrontierAfterMerge(cand)
	}
}

func (se *StreamingEncoderV2) isValidCandidate(c mergeCandidate) bool {
	i := c.leftIndex
	j := c.rightIndex

	if i < 0 || j < 0 ||
		i >= len(se.tokens) || j >= len(se.tokens) ||
		i >= len(se.next) || j >= len(se.next) ||
		i >= len(se.prev) || j >= len(se.prev) {
		return false
	}

	if se.live[i] == 0 || se.live[j] == 0 {
		return false
	}

	if se.next[i] != j || se.prev[j] != i {
		return false
	}

	if se.live[i] != c.liveLeft || se.live[j] != c.liveRight {
		return false
	}

	rank, ok := se.tok.GetPairRank(se.tokens[i], se.tokens[j])
	if !ok {
		return false
	}

	if rank != c.rank {
		return false
	}

	return true
}

func (se *StreamingEncoderV2) performMerge(c mergeCandidate) {
	i := c.leftIndex
	j := c.rightIndex

	// Note: This is called from runMerges after isValidCandidate, which already
	// validated bounds, live states, and adjacency. We trust that validation here.
	k := se.prev[i]
	l := se.next[j]

	mergedID, ok := se.tok.GetPairToken(se.tokens[i], se.tokens[j])
	if !ok {
		return
	}

	se.tokens[i] = mergedID
	se.liveGen++
	se.live[i] = se.liveGen

	se.live[j] = 0
	se.prev[j] = -1
	se.next[j] = -1

	if k != -1 {
		se.next[k] = i
	}
	se.prev[i] = k

	se.next[i] = l
	if l != -1 {
		if se.prev[l] == j {
			se.prev[l] = i
		}
	}

	if se.head == j {
		se.head = i
	}
	if se.tail == j {
		se.tail = i
	}
}

func (se *StreamingEncoderV2) updateFrontierAfterMerge(c mergeCandidate) {
	i := c.leftIndex

	k := se.prev[i]
	l := se.next[i]

	se.maybeAddCandidate(k, i)

	se.maybeAddCandidate(i, l)
}

func (se *StreamingEncoderV2) commitStablePrefix(out *[]int) {
	if se.tailReserve > 0 {
		return
	}

	se.commitPrefix(out)
}

func (se *StreamingEncoderV2) commitPrefix(out *[]int) {
	if se.head == -1 {
		return
	}

	getTokenLen := func(tokID int) int {
		if se.syntheticLengths != nil {
			if len, ok := se.syntheticLengths[tokID]; ok {
				return len
			}
		}
		return se.tok.TokenLen(tokID)
	}

	totalLen := 0
	for idx := se.head; idx != -1; idx = se.next[idx] {
		tokID := se.tokens[idx]
		totalLen += getTokenLen(tokID)
	}

	committed := 0
	var lastCommitted int = -1

	idx := se.head
	for idx != -1 {
		tokID := se.tokens[idx]
		tokLen := getTokenLen(tokID)

		remainingAfter := totalLen - (committed + tokLen)

		if remainingAfter <= se.tailReserve {
			break
		}

		*out = append(*out, tokID)
		committed += tokLen
		lastCommitted = idx

		idx = se.next[idx]
	}

	if lastCommitted == -1 {
		return
	}

	newHead := se.next[lastCommitted]

	if newHead != -1 {
		se.prev[newHead] = -1
	}

	se.head = newHead

	if newHead == -1 {
		se.tail = -1
	}
}
