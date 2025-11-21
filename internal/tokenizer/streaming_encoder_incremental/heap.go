package streaming_encoder_incremental

type mergeCandidate struct {
	leftIndex  int
	rightIndex int
	rank       int
	liveLeft   uint32
	liveRight  uint32
}

type mergeHeap struct {
	buckets    [][]mergeCandidate
	current    int
	totalCount int
}

func newMergeHeap() *mergeHeap {
	return &mergeHeap{
		buckets: make([][]mergeCandidate, 0),
		current: 0,
	}
}

func newMergeHeapWithMaxRank(maxRank int) *mergeHeap {
	return &mergeHeap{
		buckets: make([][]mergeCandidate, maxRank+1),
		current: 0,
	}
}

func (h *mergeHeap) Push(c mergeCandidate) {
	rank := c.rank
	if rank >= len(h.buckets) {
		newBuckets := make([][]mergeCandidate, rank+1)
		copy(newBuckets, h.buckets)
		h.buckets = newBuckets
	}

	h.buckets[rank] = append(h.buckets[rank], c)
	h.totalCount++

	if h.totalCount == 1 || rank < h.current {
		h.current = rank
	}
}

func (h *mergeHeap) Pop() (mergeCandidate, bool) {
	if h.totalCount == 0 {
		return mergeCandidate{}, false
	}

	for h.current < len(h.buckets) && len(h.buckets[h.current]) == 0 {
		h.current++
	}

	if h.current >= len(h.buckets) {
		h.current = 0
		return mergeCandidate{}, false
	}

	bucket := h.buckets[h.current]
	c := bucket[0]
	h.buckets[h.current] = bucket[1:]
	h.totalCount--

	return c, true
}

func (h *mergeHeap) Empty() bool {
	return h.totalCount == 0
}
