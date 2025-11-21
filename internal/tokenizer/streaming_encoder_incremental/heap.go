package streaming_encoder_incremental

type mergeCandidate struct {
	leftIndex  int
	rightIndex int
	rank       int
	liveLeft   uint32
	liveRight  uint32
}

type mergeHeap struct {
}

func newMergeHeap() *mergeHeap {
	return &mergeHeap{}
}

func (h *mergeHeap) Push(c mergeCandidate) {

}

func (h *mergeHeap) Pop() (mergeCandidate, bool) {

	return mergeCandidate{}, false
}

func (h *mergeHeap) Empty() bool {

	return true
}
