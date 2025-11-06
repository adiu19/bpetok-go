package tokenizer

import (
	"container/heap"

	"github.com/bpetok/internal/utils"
)

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
	for i := 0; i < n; i++ {
		prev[i] = i - 1
		next[i] = i + 1
	}

	// edge elements
	prev[0] = -1
	next[n-1] = -1

	// per-slot versioning to invalidate heap entries
	liveVersion := make([]int, n)

	h := &utils.MergeHeap{}
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
			heap.Push(h, utils.MergeCand{
				Rank:       rank,
				Pos:        i,
				LeftToken:  a,
				RightToken: b,
				VerL:       liveVersion[i],
				VerR:       liveVersion[j],
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
		c := heap.Pop(h).(utils.MergeCand)
		i := c.Pos
		if i == -1 {
			continue
		}

		j := next[i]
		if j == -1 {
			continue // no right neibhbor anymore
		}

		// stale entry since atleast one version did not match
		if liveVersion[i] != c.VerL || liveVersion[j] != c.VerR {
			continue
		}

		a := tokens[i]
		b := tokens[j]

		rankNow, ok := t.pairRank[[2]int{a, b}]

		// if this entry doesn’t describe the same (a,b) pair with the same rank that it did when it was pushed — skip it
		if !ok || rankNow != c.Rank || a != c.LeftToken || b != c.RightToken {
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
