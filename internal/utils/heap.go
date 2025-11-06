package utils

type MergeCand struct {
	Rank       int // lower wins
	Pos        int // left index; lower wins on tie to enforce leftmost
	LeftToken  int
	RightToken int
	VerL       int
	VerR       int
}

// MergeHeap is the type alias for our heap
type MergeHeap []MergeCand

func (h MergeHeap) Len() int { return len(h) }
func (h MergeHeap) Less(i, j int) bool {
	if h[i].Rank != h[j].Rank {
		return h[i].Rank < h[j].Rank
	}
	return h[i].Pos < h[j].Pos // leftmost tie-break
}
func (h MergeHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }
func (h *MergeHeap) Push(x any)   { *h = append(*h, x.(MergeCand)) }
func (h *MergeHeap) Pop() any     { old := *h; n := len(old); x := old[n-1]; *h = old[:n-1]; return x }
