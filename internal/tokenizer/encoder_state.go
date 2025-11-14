package tokenizer

import (
	"container/heap"
	"fmt"

	"github.com/bpetok/internal/utils"
)

// EncoderState wraps the state metadata we need for our streaming encoder
type EncoderState struct {
	tok *Tokenizer

	//max byte length we can have for a token
	lMax int

	head        int       // index of the first live node in nodes
	nodes       []encNode // list of nodes, where each node is a token
	liveVersion int       // bump to invalidate stale pairs

	// index into nodes ( >= hed). everything strictly before this index is finalized and can be emitted
	sealIdx int

	outBuf []int // finalized token ids ready to be emitted

	// min-heap by rank; pairs reference nodes via indices and version
	heap utils.MergeHeap
}

type encNode struct {
	id          int  // current token id
	left, right int  // neighboring indices or -1 if no neighbors
	ver         int  // node version
	byteLen     int  // cached length in bytes of token (sum of subparts)
	dead        bool // suggests that the node has been delinked and is no longer valid
}

func (st *EncoderState) bump() int {
	st.liveVersion++
	return st.liveVersion
}

// NewEncoderState returns a new instance of the encoder state
func NewEncoderState(t *Tokenizer) *EncoderState {
	encState := &EncoderState{tok: t}
	encState.lMax = t.MaxTokenByteLen
	encState.head, encState.sealIdx = 0, 0
	heap.Init(&encState.heap)
	return encState
}

// appendByte creates a 1-byte token node and links with previous tail node
func (st *EncoderState) appendByte(b byte) {
	id := st.tok.byteToToken[b]
	i := len(st.nodes)
	left := i - 1
	st.nodes = append(st.nodes, encNode{
		id:      id,
		left:    left,
		right:   -1,
		ver:     st.liveVersion,
		byteLen: 1,
	})

	if left >= 0 {
		st.nodes[left].right = i
		st.enqueuePair(left, i)
	}
}

// enqueuePair enqueues the adjacent pair if it exists in our pairRank built during tokenizer load
func (st *EncoderState) enqueuePair(l, r int) {
	if l < 0 || r < 0 {
		return
	}

	rank, ok := st.tok.pairRank[[2]int{st.nodes[l].id, st.nodes[r].id}]
	if !ok {
		fmt.Printf("unable to fetch pair rank for left_index =  %d and right_index = %d ", l, r)
		return
	}

	heap.Push(&st.heap, utils.MergeCand{
		Rank:       rank,
		Pos:        l,
		LeftToken:  st.nodes[l].id,
		RightToken: st.nodes[r].id,
		VerL:       st.nodes[l].ver,
		VerR:       st.nodes[r].ver,
	})

}

// mergeAt replaces left + right with merged token on the left node
// it also updates neighbors, bumps versions, and enqueues new bordering pairs.
func (st *EncoderState) mergeAt(l, r, mergedID int) {
	L := &st.nodes[l]
	R := &st.nodes[r]

	L.id = mergedID
	L.byteLen += R.byteLen
	L.ver = st.bump()

	// link nodes
	nr := R.right
	L.right = nr
	if nr >= 0 {
		st.nodes[nr].left = l
		st.nodes[nr].ver = st.bump()
		st.nodes[nr].right = nr
		st.enqueuePair(l, nr)
	}

	ll := L.left
	if ll >= 0 {
		st.nodes[ll].ver = st.bump()
		st.enqueuePair(ll, l)
	}

	R.dead = true
}

// tailStartIdx returns index of first node in the kept tail window
func (st *EncoderState) tailStartIdx() int {
	last := st.lastLive()
	if last == -1 {
		return st.sealIdx
	}

	budget := st.lMax - 1
	sum := 0
	i := last // start from the rightmost node that is live
	for i >= st.sealIdx {
		if !st.nodes[i].dead {
			if sum+st.nodes[i].byteLen > budget {
				break
			}
			sum += st.nodes[i].byteLen
		}

		i = st.nodes[i].left // walk left
	}

	if i < st.sealIdx {
		return st.sealIdx // stop at sealIdx since that's the start of our tail
	}

	return st.nodes[i].right // we overflowed our bytes at i, which means we need to consider the right node
}

func (st *EncoderState) lastLive() int {
	for i := len(st.nodes) - 1; i >= st.sealIdx; i-- {
		if !st.nodes[i].dead {
			return i
		}
	}
	return -1
}

// advance sealIdx upto tail start, emitting tokens in the process
func (st *EncoderState) sealCommitted() {
	tailStart := st.tailStartIdx()
	for st.sealIdx >= 0 && st.sealIdx < tailStart {
		n := &st.nodes[st.sealIdx]
		if !n.dead {
			st.outBuf = append(st.outBuf, n.id)
		}

		st.sealIdx++
		st.head = st.sealIdx // monotonic
	}
}

// Push
func (st *EncoderState) Push(chunk []byte) []int {
	st.outBuf = st.outBuf[:0]
	for _, b := range chunk {
		st.appendByte(b)
	}

	for st.heap.Len() > 0 {
		cand := heap.Pop(&st.heap).(utils.MergeCand)

		l := cand.Pos
		if l < 0 || l >= len(st.nodes) {
			continue
		}

		r := st.nodes[l].right
		if r < 0 {
			continue
		}

		// stale?
		if st.nodes[l].ver != cand.VerL || st.nodes[r].ver != cand.VerR {
			continue
		}

		// tokens changed?
		if st.nodes[l].id != cand.LeftToken || st.nodes[r].id != cand.RightToken {
			continue
		}

		// still a valid merge?
		mergedID, ok := st.tok.pairToken[[2]int{st.nodes[l].id, st.nodes[r].id}]
		if !ok {
			continue
		}

		st.mergeAt(l, r, mergedID)
	}

	st.sealCommitted()

	out := make([]int, len(st.outBuf))
	copy(out, st.outBuf)
	return out
}

// Flush whatever remains
func (st *EncoderState) Flush() []int {
	out := st.outBuf[:0]
	for i := st.head; i <= len(st.nodes); i++ {
		if !st.nodes[i].dead {
			out = append(out, st.nodes[i].id)
		}
	}

	st.nodes = st.nodes[:0]
	st.head, st.sealIdx = 0, 0
	st.heap = st.heap[:0]
	return append([]int(nil), out...)
}
