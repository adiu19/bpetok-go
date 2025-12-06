package core

import (
	"github.com/bpetok/internal/utils"
)

type BaseEncoderState struct {
}

func (t *Tokenizer) EncodeOffline(input []byte, state *BaseEncoderState) []int {
	n := len(input)
	if n == 0 {
		return nil
	}

	scratch := t.acquireScratch(n)
	defer t.releaseScratch(scratch)

	tokens := scratch.tokens

	for i, b := range input {
		tokens[i] = t.byteToToken[b]
	}

	// doubly linked-list
	prev := scratch.prev
	next := scratch.next
	for i := 0; i < n; i++ {
		prev[i] = i - 1
		next[i] = i + 1
	}

	// edge elements
	prev[0] = -1
	next[n-1] = -1

	liveVersion := scratch.live
	for i := 0; i < n; i++ {
		liveVersion[i] = 0
	}

	h := utils.NewBucketQueue(t.maxRank)

	pushIfMergeable := func(i int) {
		j := next[i]
		if i == -1 || j == -1 {
			return
		}

		a := tokens[i]
		b := tokens[j]

		info, ok := t.pairLookup.Lookup(a, b)
		if ok {
			rank := int(info >> 32)
			h.Push(utils.MergeCand{
				Rank:       rank,
				Pos:        i,
				LeftToken:  a,
				RightToken: b,
				VerL:       liveVersion[i],
				VerR:       liveVersion[j],
			})
		}
	}

	for i := 0; i != -1 && next[i] != -1; i = next[i] {
		pushIfMergeable(i)
	}

	// leftmost index (never dies; we always merge into the left slot)
	head := 0

	for {
		c, ok := h.Pop()
		if !ok {
			break
		}
		i := c.Pos
		if i == -1 {
			continue
		}

		j := next[i]
		if j == -1 {
			continue
		}

		if liveVersion[i] != c.VerL || liveVersion[j] != c.VerR {
			continue
		}

		a := tokens[i]
		b := tokens[j]

		info, ok := t.pairLookup.Lookup(a, b)
		if !ok {
			continue
		}

		rankNow := int(info >> 32)
		cID := int(info & 0xFFFFFFFF)

		if rankNow != c.Rank || a != c.LeftToken || b != c.RightToken {
			continue
		}

		tokens[i] = cID

		nj := next[j]
		next[i] = nj
		if nj != -1 {
			prev[nj] = i
		}

		prev[j], next[j] = -1, -1

		liveVersion[i]++
		liveVersion[j]++

		if pi := prev[i]; pi != -1 {
			pushIfMergeable(pi)
		}

		pushIfMergeable(i)
	}

	out := make([]int, 0, n)
	for i := head; i != -1; i = next[i] {
		out = append(out, tokens[i])
	}

	return out
}

type encodeScratch struct {
	tokens []int
	prev   []int
	next   []int
	live   []int
}

func (t *Tokenizer) acquireScratch(n int) *encodeScratch {
	v := t.scratchPool.Get()
	var sc *encodeScratch
	if v == nil {
		sc = &encodeScratch{}
	} else {
		sc = v.(*encodeScratch)
	}
	sc.prepare(n)
	return sc
}

func (t *Tokenizer) releaseScratch(sc *encodeScratch) {
	t.scratchPool.Put(sc)
}

func (sc *encodeScratch) prepare(n int) {
	sc.tokens = ensureIntCapacity(sc.tokens, n)
	sc.prev = ensureIntCapacity(sc.prev, n)
	sc.next = ensureIntCapacity(sc.next, n)
	sc.live = ensureIntCapacity(sc.live, n)
}

func ensureIntCapacity(buf []int, n int) []int {
	if cap(buf) < n {
		return make([]int, n)
	}
	return buf[:n]
}
