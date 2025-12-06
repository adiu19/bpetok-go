package utils

const (
	defaultHeapPrealloc = 8192
)

type MergeQueue interface {
	Push(c MergeCand)
	Pop() (MergeCand, bool)
	Len() int
	Reset()
}

type MergeHeap struct {
	items           []MergeCand
	preAllocated    bool
	initialCapacity int
}

func NewMergeHeap(preAlloc ...bool) *MergeHeap {
	shouldPreAlloc := false
	if len(preAlloc) > 0 {
		shouldPreAlloc = preAlloc[0]
	}

	var items []MergeCand
	var initialCap int
	if shouldPreAlloc {
		initialCap = defaultHeapPrealloc
		items = make([]MergeCand, 0, defaultHeapPrealloc)
	} else {
		initialCap = 64
		items = make([]MergeCand, 0, 64)
	}

	return &MergeHeap{
		items:           items,
		preAllocated:    shouldPreAlloc,
		initialCapacity: initialCap,
	}
}

func (h *MergeHeap) Len() int {
	return len(h.items)
}

func (h *MergeHeap) less(a, b MergeCand) bool {
	if a.Rank != b.Rank {
		return a.Rank < b.Rank
	}
	return a.Pos < b.Pos
}

func (h *MergeHeap) Push(c MergeCand) {
	h.items = append(h.items, c)
	h.up(len(h.items) - 1)
}

func (h *MergeHeap) Pop() (MergeCand, bool) {
	if len(h.items) == 0 {
		return MergeCand{}, false
	}

	n := len(h.items) - 1
	h.items[0], h.items[n] = h.items[n], h.items[0]

	result := h.items[n]
	h.items = h.items[:n]

	if len(h.items) > 0 {
		h.down(0)
	}

	return result, true
}

func (h *MergeHeap) up(i int) {
	for {
		parent := (i - 1) / 2
		if parent == i || !h.less(h.items[i], h.items[parent]) {
			break
		}
		h.items[parent], h.items[i] = h.items[i], h.items[parent]
		i = parent
	}
}

func (h *MergeHeap) down(i int) {
	n := len(h.items)
	for {
		left := 2*i + 1
		right := 2*i + 2
		smallest := i

		if left < n && h.less(h.items[left], h.items[smallest]) {
			smallest = left
		}
		if right < n && h.less(h.items[right], h.items[smallest]) {
			smallest = right
		}
		if smallest == i {
			break
		}
		h.items[i], h.items[smallest] = h.items[smallest], h.items[i]
		i = smallest
	}
}

func (h *MergeHeap) Reset() {
	if h.preAllocated {
		h.items = h.items[:0]
	} else {
		h.items = nil
	}
}
