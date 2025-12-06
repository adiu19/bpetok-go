package streaming_encoder_incremental

import (
	"github.com/bpetok/internal/tokenizer/core"
	"math/rand"
	"reflect"
	"testing"
)

type mockHeap struct {
	items []mergeCandidate
}

func (h *mockHeap) Push(c mergeCandidate) {

	for _, existing := range h.items {
		if existing.leftIndex == c.leftIndex && existing.rightIndex == c.rightIndex {
			return
		}
	}
	h.items = append(h.items, c)
}

func (h *mockHeap) Pop() (mergeCandidate, bool) { return mergeCandidate{}, false }
func (h *mockHeap) Empty() bool                 { return len(h.items) == 0 }
func (h *mockHeap) Reset() {
	h.items = h.items[:0]
}

// -----------------------

// spyHeapForFrontierTest wraps an existing heap to spy on Push calls
type spyHeapForFrontierTest struct {
	wrapped heapInterface
	onPush  func(mergeCandidate)
}

func (s *spyHeapForFrontierTest) Push(c mergeCandidate) {
	if s.onPush != nil {
		s.onPush(c)
	}
	s.wrapped.Push(c)
}

func (s *spyHeapForFrontierTest) Pop() (mergeCandidate, bool) {
	return s.wrapped.Pop()
}

func (s *spyHeapForFrontierTest) Empty() bool {
	return s.wrapped.Empty()
}

func (s *spyHeapForFrontierTest) Reset() {
	s.wrapped.Reset()
}

// ----------------------

func setupTwoNodeEncoder(t *testing.T, left, right byte) (*StreamingEncoderV2, int, int) {
	tok, err := core.LoadTokenizerFromFiles("../testdata/gpt2/vocab.json", "../testdata/gpt2/merges.txt")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}

	se := NewStreamingEncoderV2(tok)
	se.heap = &mockHeap{}

	nodes := se.appendBytes([]byte{left, right})
	if len(nodes) != 2 {
		t.Fatalf("expected 2 nodes, got %v", nodes)
	}

	return se, nodes[0], nodes[1]
}

func TestPerformMerge_MiddleOfList(t *testing.T) {
	tok, err := core.LoadTokenizerFromFiles("../testdata/gpt2/vocab.json", "../testdata/gpt2/merges.txt")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}

	se := NewStreamingEncoderV2(tok)

	indices := newSyntheticList(se, []int{1, 1, 1})
	se.tokens[indices[0]] = tok.GetByteToToken('h')
	se.tokens[indices[1]] = tok.GetByteToToken('e')
	se.tokens[indices[2]] = tok.GetByteToToken('l')

	left := indices[0]
	right := indices[1]

	mergedID, ok := tok.GetPairToken(se.tokens[left], se.tokens[right])
	if !ok {
		t.Fatalf("test assumption: 'h' 'e' must be mergeable")
	}

	cand := mergeCandidate{
		leftIndex:  left,
		rightIndex: right,
		rank:       0,
		liveLeft:   se.live[left],
		liveRight:  se.live[right],
	}

	se.performMerge(cand)

	if se.tokens[left] != mergedID {
		t.Fatalf("left node should contain merged tokenID %d, got %d", mergedID, se.tokens[left])
	}

	if se.live[right] != 0 {
		t.Fatalf("right node should be dead (live=0), got %d", se.live[right])
	}

	if se.next[left] != indices[2] {
		t.Fatalf("expected merged node next=%d, got %d", indices[2], se.next[left])
	}
	if se.prev[indices[2]] != left {
		t.Fatalf("expected C.prev = left")
	}

	if se.head != left {
		t.Fatalf("expected head=%d got %d", left, se.head)
	}
}

func TestPerformMerge_AtTail(t *testing.T) {
	tok, err := core.LoadTokenizerFromFiles("../testdata/gpt2/vocab.json", "../testdata/gpt2/merges.txt")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}

	se := NewStreamingEncoderV2(tok)

	indices := newSyntheticList(se, []int{1, 1})
	se.tokens[indices[0]] = tok.GetByteToToken('l')
	se.tokens[indices[1]] = tok.GetByteToToken('o')

	left := indices[0]
	right := indices[1]

	mergedID, ok := tok.GetPairToken(se.tokens[left], se.tokens[right])
	if !ok {
		t.Fatalf("test assumption: 'l' 'o' must be mergeable")
	}

	cand := mergeCandidate{
		leftIndex:  left,
		rightIndex: right,
		liveLeft:   se.live[left],
		liveRight:  se.live[right],
	}

	se.performMerge(cand)

	if se.tokens[left] != mergedID {
		t.Fatalf("expected merged token=%d, got %d", mergedID, se.tokens[left])
	}

	if se.live[right] != 0 {
		t.Fatalf("expected right node dead after merge")
	}

	if se.tail != left {
		t.Fatalf("expected tail=%d, got %d", left, se.tail)
	}

	if se.next[left] != -1 {
		t.Fatalf("expected merged tail to have next=-1")
	}
}

func TestPerformMerge_TokenIdCorrectness(t *testing.T) {
	tok, err := core.LoadTokenizerFromFiles("../testdata/gpt2/vocab.json", "../testdata/gpt2/merges.txt")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}

	se := NewStreamingEncoderV2(tok)
	indices := newSyntheticList(se, []int{1, 1})

	se.tokens[indices[0]] = tok.GetByteToToken('e')
	se.tokens[indices[1]] = tok.GetByteToToken('r')

	mergedID, ok := tok.GetPairToken(se.tokens[indices[0]], se.tokens[indices[1]])
	if !ok {
		t.Fatalf("'er' must be mergeable")
	}

	se.performMerge(mergeCandidate{
		leftIndex:  indices[0],
		rightIndex: indices[1],
		liveLeft:   se.live[indices[0]],
		liveRight:  se.live[indices[1]],
	})

	if se.tokens[indices[0]] != mergedID {
		t.Fatalf("expected merged token %d, got %d", mergedID, se.tokens[indices[0]])
	}
}

func TestPerformMerge_NoOverDelete(t *testing.T) {
	tok, _ := core.LoadTokenizerFromFiles("../testdata/gpt2/vocab.json", "../testdata/gpt2/merges.txt")
	se := NewStreamingEncoderV2(tok)

	indices := newSyntheticList(se, []int{1, 1, 1})

	se.tokens[indices[0]] = tok.GetByteToToken('t')
	se.tokens[indices[1]] = tok.GetByteToToken('h')
	se.tokens[indices[2]] = tok.GetByteToToken('e')

	_, ok := tok.GetPairToken(se.tokens[indices[0]], se.tokens[indices[1]])
	if !ok {
		t.Skip("GPT2 merges might differ; skipping test if 'th' is not mergeable")
	}

	se.performMerge(mergeCandidate{
		leftIndex:  indices[0],
		rightIndex: indices[1],
		liveLeft:   se.live[indices[0]],
		liveRight:  se.live[indices[1]],
	})

	if se.live[indices[2]] == 0 {
		t.Fatalf("right neighbor incorrectly removed")
	}
	if se.live[indices[0]] == 0 {
		t.Fatalf("left node should stay alive")
	}
	if se.live[indices[1]] != 0 {
		t.Fatalf("middle node should be dead")
	}
}

func TestAppendBytes_ListIntegrity(t *testing.T) {
	tok, err := core.LoadTokenizerFromFiles("../testdata/gpt2/vocab.json", "../testdata/gpt2/merges.txt")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}

	se := NewStreamingEncoderV2(tok)

	chunk1 := []byte("hel")
	nodes1 := se.appendBytes(chunk1)

	if !reflect.DeepEqual(nodes1, []int{0, 1, 2}) {
		t.Fatalf("nodes1 mismatch: got %v", nodes1)
	}

	wantNext1 := []int{1, 2, -1}
	wantPrev1 := []int{-1, 0, 1}
	for i := 0; i < 3; i++ {
		if se.next[i] != wantNext1[i] {
			t.Fatalf("next[%d] = %d, want %d", i, se.next[i], wantNext1[i])
		}
		if se.prev[i] != wantPrev1[i] {
			t.Fatalf("prev[%d] = %d, want %d", i, se.prev[i], wantPrev1[i])
		}
	}
	if se.head != 0 || se.tail != 2 {
		t.Fatalf("wrong head/tail after chunk1: head=%d tail=%d", se.head, se.tail)
	}

	chunk2 := []byte("lo")
	nodes2 := se.appendBytes(chunk2)

	if !reflect.DeepEqual(nodes2, []int{3, 4}) {
		t.Fatalf("nodes2 mismatch: got %v", nodes2)
	}

	wantNext2 := []int{1, 2, 3, 4, -1}
	wantPrev2 := []int{-1, 0, 1, 2, 3}

	for i := 0; i < 5; i++ {
		if se.next[i] != wantNext2[i] {
			t.Fatalf("after chunk2: next[%d] = %d, want %d", i, se.next[i], wantNext2[i])
		}
		if se.prev[i] != wantPrev2[i] {
			t.Fatalf("after chunk2: prev[%d] = %d, want %d", i, se.prev[i], wantPrev2[i])
		}
	}
	if se.head != 0 || se.tail != 4 {
		t.Fatalf("wrong head/tail after chunk2: head=%d tail=%d", se.head, se.tail)
	}

	wantTokens := []int{
		tok.GetByteToToken('h'),
		tok.GetByteToToken('e'),
		tok.GetByteToToken('l'),
		tok.GetByteToToken('l'),
		tok.GetByteToToken('o'),
	}

	gotTokens := []int{
		se.tokens[0],
		se.tokens[1],
		se.tokens[2],
		se.tokens[3],
		se.tokens[4],
	}

	if !reflect.DeepEqual(gotTokens, wantTokens) {
		t.Fatalf("token mismatch:\n got  %v\n want %v", gotTokens, wantTokens)
	}
}

func TestCommitPrefix_NoCommit(t *testing.T) {
	tok, err := core.LoadTokenizerFromFiles("../testdata/gpt2/vocab.json", "../testdata/gpt2/merges.txt")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}

	se := NewStreamingEncoderV2(tok)

	newSyntheticList(se, []int{1, 1, 1})
	se.tailReserve = 2

	out := []int{}
	se.commitPrefix(&out)

	if len(out) != 0 {
		t.Fatalf("expected no committed tokens, got %d", len(out))
	}
	if se.head == -1 {
		t.Fatalf("head must not move")
	}
}

func TestCommitPrefix_CommitOne(t *testing.T) {
	tok, err := core.LoadTokenizerFromFiles("../testdata/gpt2/vocab.json", "../testdata/gpt2/merges.txt")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}

	se := NewStreamingEncoderV2(tok)

	indices := newSyntheticList(se, []int{1, 1, 1, 1})
	se.tailReserve = 2

	out := []int{}
	se.commitPrefix(&out)

	if len(out) != 1 {
		t.Fatalf("expected 1 committed token, got %d", len(out))
	}
	if out[0] != 0 {
		t.Fatalf("expected committed tokenID = 0, got %d", out[0])
	}
	if se.head != indices[1] {
		t.Fatalf("expected head to move to second node")
	}
}

func TestCommitPrefix_CommitMultiple(t *testing.T) {
	tok, err := core.LoadTokenizerFromFiles("../testdata/gpt2/vocab.json", "../testdata/gpt2/merges.txt")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}

	se := NewStreamingEncoderV2(tok)

	indices := newSyntheticList(se, []int{1, 1, 1, 1, 1})
	se.tailReserve = 2

	out := []int{}
	se.commitPrefix(&out)

	if len(out) != 2 {
		t.Fatalf("expected 2 committed tokens, got %d", len(out))
	}

	want := []int{0, 1}
	if !reflect.DeepEqual(out, want) {
		t.Fatalf("wrong committed token order: got %v want %v", out, want)
	}

	if se.head != indices[2] {
		t.Fatalf("expected head to advance to index 3")
	}
}

func TestCommitPrefix_CommitAllOnFlush(t *testing.T) {
	tok, err := core.LoadTokenizerFromFiles("../testdata/gpt2/vocab.json", "../testdata/gpt2/merges.txt")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}

	se := NewStreamingEncoderV2(tok)

	_ = newSyntheticList(se, []int{1, 1, 1})

	se.tailReserve = 0

	out := []int{}
	se.commitPrefix(&out)

	if len(out) != 2 {
		t.Fatalf("expected 2 committed tokens, got %d", len(out))
	}

	if se.head == -1 {
		t.Fatalf("head should NOT be empty after commitPrefix")
	}
}

func TestUpdateFrontierAfterMerge_LeftNeighborMergeable(t *testing.T) {
	tok, err := core.LoadTokenizerFromFiles("../testdata/gpt2/vocab.json", "../testdata/gpt2/merges.txt")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	se := NewStreamingEncoderV2(tok)
	mh := &mockHeap{}
	se.heap = mh

	nodes := se.appendBytes([]byte("the"))
	k := nodes[0]
	i := nodes[1]
	j := nodes[2]

	rank, ok := tok.GetPairRank(se.tokens[i], se.tokens[j])
	if !ok {
		t.Fatalf("'h','e' must be mergeable")
	}

	c := mergeCandidate{i, j, rank, se.live[i], se.live[j]}
	se.performMerge(c)

	se.updateFrontierAfterMerge(c)

	if len(mh.items) != 1 {
		t.Fatalf("expected 1 candidate, got %d", len(mh.items))
	}

	cand := mh.items[0]
	if cand.leftIndex != k || cand.rightIndex != i {
		t.Fatalf("expected push of (k,i)=(%d,%d), got (%d,%d)",
			k, i, cand.leftIndex, cand.rightIndex)
	}
}

func TestUpdateFrontierAfterMerge_RightNeighborMergeable(t *testing.T) {
	tok, err := core.LoadTokenizerFromFiles("../testdata/gpt2/vocab.json", "../testdata/gpt2/merges.txt")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}

	se := NewStreamingEncoderV2(tok)
	mh := &mockHeap{}
	se.heap = mh

	nodes := se.appendBytes([]byte("ings"))
	if len(nodes) != 4 {
		t.Fatalf("expected 4 nodes for 'ings'")
	}

	idxI := nodes[0]
	idxN := nodes[1]
	idxG := nodes[2]
	idxS := nodes[3]

	rankIN, ok := tok.GetPairRank(se.tokens[idxI], se.tokens[idxN])
	if !ok {
		t.Fatalf("'i','n' must be mergeable (forming 'in')")
	}

	cIN := mergeCandidate{
		leftIndex:  idxI,
		rightIndex: idxN,
		rank:       rankIN,
		liveLeft:   se.live[idxI],
		liveRight:  se.live[idxN],
	}
	se.performMerge(cIN)

	if se.next[idxI] != idxG {
		t.Fatalf("expected next of merged 'in' to be g")
	}

	rankING, ok := tok.GetPairRank(se.tokens[idxI], se.tokens[idxG])
	if !ok {
		t.Fatalf("'in','g' must be mergeable (forming 'ing')")
	}

	cING := mergeCandidate{
		leftIndex:  idxI,
		rightIndex: idxG,
		rank:       rankING,
		liveLeft:   se.live[idxI],
		liveRight:  se.live[idxG],
	}
	se.performMerge(cING)

	if se.next[idxI] != idxS {
		t.Fatalf("expected next of merged 'ing' to be s")
	}

	_, ok = tok.GetPairRank(se.tokens[idxI], se.tokens[idxS])
	if !ok {
		t.Fatalf("'ing','s' must be mergeable (forming 'ings')")
	}
	se.updateFrontierAfterMerge(cING)

	found := false
	for _, cand := range mh.items {
		if cand.leftIndex == idxI && cand.rightIndex == idxS {
			found = true
			break
		}
	}

	if !found {
		t.Fatalf("expected (idxI,idxS) = ('ing','s') to be pushed to heap")
	}
}

func TestUpdateFrontierAfterMerge_NoPushIfNotMergeable(t *testing.T) {
	tok, err := core.LoadTokenizerFromFiles("../testdata/gpt2/vocab.json", "../testdata/gpt2/merges.txt")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	se := NewStreamingEncoderV2(tok)
	mh := &mockHeap{}
	se.heap = mh

	nodes := se.appendBytes([]byte("er!"))
	i := nodes[0]
	j := nodes[1]
	l := nodes[2]

	rank, ok := tok.GetPairRank(se.tokens[i], se.tokens[j])
	if !ok {
		t.Fatalf("'e','r' must be mergeable")
	}

	c := mergeCandidate{i, j, rank, se.live[i], se.live[j]}
	se.performMerge(c)

	_, ok = tok.GetPairRank(se.tokens[i], se.tokens[l])
	if ok {
		t.Fatalf("test assumption broken: 'er','!' must NOT merge")
	}

	se.updateFrontierAfterMerge(c)

	if len(mh.items) != 0 {
		t.Fatalf("expected no pushes, got %d", len(mh.items))
	}
}

func TestUpdateFrontierAfterMerge_LiveVersionCaptured(t *testing.T) {
	tok, err := core.LoadTokenizerFromFiles("../testdata/gpt2/vocab.json", "../testdata/gpt2/merges.txt")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	se := NewStreamingEncoderV2(tok)
	mh := &mockHeap{}
	se.heap = mh

	nodes := se.appendBytes([]byte("the"))
	k := nodes[0]
	i := nodes[1]
	j := nodes[2]

	rank, _ := tok.GetPairRank(se.tokens[i], se.tokens[j])
	c := mergeCandidate{i, j, rank, se.live[i], se.live[j]}

	se.performMerge(c)

	liveK := se.live[k]
	liveI := se.live[i]

	se.updateFrontierAfterMerge(c)

	if len(mh.items) != 1 {
		t.Fatalf("expected 1 heap push")
	}

	cand := mh.items[0]
	if cand.liveLeft != liveK || cand.liveRight != liveI {
		t.Fatalf("expected live versions (%d,%d), got (%d,%d)",
			liveK, liveI, cand.liveLeft, cand.liveRight)
	}
}

func TestUpdateFrontierAfterMerge_NoUnexpectedDuplicates(t *testing.T) {
	tok, err := core.LoadTokenizerFromFiles("../testdata/gpt2/vocab.json", "../testdata/gpt2/merges.txt")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	se := NewStreamingEncoderV2(tok)
	mh := &mockHeap{}
	se.heap = mh

	nodes := se.appendBytes([]byte("the"))
	k := nodes[0]
	i := nodes[1]
	j := nodes[2]

	rank, _ := tok.GetPairRank(se.tokens[i], se.tokens[j])
	c := mergeCandidate{i, j, rank, se.live[i], se.live[j]}

	se.performMerge(c)

	se.updateFrontierAfterMerge(c)
	se.updateFrontierAfterMerge(c)

	if len(mh.items) != 1 {
		t.Fatalf("expected exactly 1 push, got %d", len(mh.items))
	}

	cand := mh.items[0]
	if cand.leftIndex != k || cand.rightIndex != i {
		t.Fatalf("expected candidate (k,i)")
	}
}

func TestPerformMerge_Simple(t *testing.T) {
	se, i, j := setupTwoNodeEncoder(t, 'e', 'r')

	leftTok := se.tokens[i]
	rightTok := se.tokens[j]
	mergedID, ok := se.tok.GetPairToken(leftTok, rightTok)
	if !ok {
		t.Fatalf("test assumption: 'er' should be mergeable")
	}

	c := mergeCandidate{
		leftIndex:  i,
		rightIndex: j,
		rank:       0,
		liveLeft:   se.live[i],
		liveRight:  se.live[j],
	}

	se.performMerge(c)

	if se.tokens[i] != mergedID {
		t.Fatalf("tokens[%d] = %d, expected mergedID %d", i, se.tokens[i], mergedID)
	}

	if se.prev[j] != -1 || se.next[j] != -1 {
		t.Fatalf("node j should be isolated")
	}

	if se.prev[i] != -1 {
		t.Fatalf("prev of merged node i should be -1 (it was head)")
	}
	if se.next[i] != -1 {
		t.Fatalf("next of merged node i should be -1")
	}
}

func TestPerformMerge_LiveVersionUpdated(t *testing.T) {
	se, i, j := setupTwoNodeEncoder(t, 'a', 'b')

	lvBefore := se.live[i]

	leftTok := se.tokens[i]
	rightTok := se.tokens[j]
	mergedID, _ := se.tok.GetPairToken(leftTok, rightTok)

	c := mergeCandidate{
		leftIndex:  i,
		rightIndex: j,
		rank:       0,
		liveLeft:   se.live[i],
		liveRight:  se.live[j],
	}

	se.performMerge(c)

	if se.live[i] == lvBefore {
		t.Fatalf("expected live version to increase")
	}
	if se.tokens[i] == leftTok {
		t.Fatalf("expected merged token, got original %d", leftTok)
	}
	if se.tokens[i] != mergedID {
		t.Fatalf("expected merged token %d, got %d", mergedID, se.tokens[i])
	}
}

func TestPerformMerge_MiddleThreeNodes(t *testing.T) {
	tok, err := core.LoadTokenizerFromFiles("../testdata/gpt2/vocab.json", "../testdata/gpt2/merges.txt")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	se := NewStreamingEncoderV2(tok)
	se.heap = &mockHeap{}

	nodes := se.appendBytes([]byte("abc"))
	k, i, j, l := nodes[0], nodes[1], nodes[1], nodes[2]

	se = NewStreamingEncoderV2(tok)
	nodes = se.appendBytes([]byte("abcd"))
	k = nodes[0]
	i = nodes[1]
	j = nodes[2]
	l = nodes[3]

	leftTok := se.tokens[i]
	rightTok := se.tokens[j]
	mergedID, ok := tok.GetPairToken(leftTok, rightTok)
	if !ok {
		t.Fatalf("test assumption: 'i','j' must be mergeable in GPT2")
	}

	c := mergeCandidate{
		leftIndex:  i,
		rightIndex: j,
		rank:       0,
		liveLeft:   se.live[i],
		liveRight:  se.live[j],
	}

	se.performMerge(c)

	if se.prev[i] != k {
		t.Fatalf("expected prev[i] = k (%d), got %d", k, se.prev[i])
	}
	if se.next[i] != l {
		t.Fatalf("expected next[i] = l (%d), got %d", l, se.next[i])
	}

	if se.prev[j] != -1 || se.next[j] != -1 {
		t.Fatalf("j should be isolated")
	}

	if se.next[k] != i {
		t.Fatalf("k.next = %d, want %d", se.next[k], i)
	}
	if se.prev[l] != i {
		t.Fatalf("l.prev = %d, want %d", se.prev[l], i)
	}

	if se.tokens[i] != mergedID {
		t.Fatalf("expected merged token %d at i, got %d", mergedID, se.tokens[i])
	}
}

func TestPerformMerge_UpdatesHead(t *testing.T) {
	se, i, j := setupTwoNodeEncoder(t, 'x', 'y')

	leftTok := se.tokens[i]
	rightTok := se.tokens[j]
	mergedID, _ := se.tok.GetPairToken(leftTok, rightTok)

	c := mergeCandidate{
		leftIndex:  i,
		rightIndex: j,
		rank:       0,
		liveLeft:   se.live[i],
		liveRight:  se.live[j],
	}

	se.performMerge(c)

	if se.head != i {
		t.Fatalf("expected head to be i after merge, got %d", se.head)
	}
	if se.tokens[se.head] != mergedID {
		t.Fatalf("expected merged token ID at head")
	}
}

func TestPerformMerge_UpdatesTail(t *testing.T) {
	tok, err := core.LoadTokenizerFromFiles("../testdata/gpt2/vocab.json", "../testdata/gpt2/merges.txt")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	se := NewStreamingEncoderV2(tok)
	se.heap = &mockHeap{}

	nodes := se.appendBytes([]byte("ab"))
	i, j := nodes[0], nodes[1]

	leftTok := se.tokens[i]
	rightTok := se.tokens[j]
	mergedID, _ := tok.GetPairToken(leftTok, rightTok)

	c := mergeCandidate{
		leftIndex:  i,
		rightIndex: j,
		rank:       0,
		liveLeft:   se.live[i],
		liveRight:  se.live[j],
	}

	se.performMerge(c)

	if se.tail != i {
		t.Fatalf("tail should now be i, got %d", se.tail)
	}
	if se.tokens[i] != mergedID {
		t.Fatalf("merged token incorrect at tail")
	}
}

func TestPerformMerge_MultipleMerges(t *testing.T) {
	tok, err := core.LoadTokenizerFromFiles("../testdata/gpt2/vocab.json", "../testdata/gpt2/merges.txt")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	se := NewStreamingEncoderV2(tok)
	se.heap = &mockHeap{}

	testSequences := [][]byte{
		[]byte("er "),
		[]byte("er."),
		[]byte("er,"),
		[]byte("the"),
		[]byte("ing"),
	}

	var found bool
	for _, seq := range testSequences {
		if len(seq) < 3 {
			continue
		}
		se = NewStreamingEncoderV2(tok)
		se.heap = &mockHeap{}

		nodes := se.appendBytes(seq)
		i1, j1 := nodes[0], nodes[1]
		i2, j2 := i1, nodes[2]

		rank1, ok1 := se.tok.GetPairRank(se.tokens[i1], se.tokens[j1])
		if !ok1 {
			continue
		}

		c1 := mergeCandidate{i1, j1, rank1, se.live[i1], se.live[j1]}
		se.performMerge(c1)

		if se.next[i1] != nodes[2] {
			continue
		}

		rank2, ok2 := se.tok.GetPairRank(se.tokens[i2], se.tokens[j2])
		if !ok2 {
			continue
		}

		found = true

		c2 := mergeCandidate{i2, j2, rank2, se.live[i2], se.live[j2]}
		se.performMerge(c2)

		if se.next[i2] != -1 {
			t.Fatalf("after second merge, next[i] should be -1 (tail), got %d", se.next[i2])
		}
		if se.tail != i2 {
			t.Fatalf("after second merge, tail should be i2 (%d), got %d", i2, se.tail)
		}
		break
	}

	if !found {
		t.Skip("could not find a sequence where both merges are possible in GPT-2")
	}
}

func TestIsValidCandidate_Valid(t *testing.T) {
	se, i, j := setupTwoNodeEncoder(t, 'e', 'r')

	tok := se.tok
	leftTok := se.tokens[i]
	rightTok := se.tokens[j]
	rank, ok := tok.GetPairRank(leftTok, rightTok)
	if !ok {
		t.Fatalf("test assumption broken: 'er' must be mergeable in GPT-2")
	}

	cand := mergeCandidate{
		leftIndex:  i,
		rightIndex: j,
		rank:       rank,
		liveLeft:   se.live[i],
		liveRight:  se.live[j],
	}

	if !se.isValidCandidate(cand) {
		t.Fatalf("expected candidate to be valid but got invalid")
	}
}

func TestIsValidCandidate_LiveVersionMismatch(t *testing.T) {
	se, i, j := setupTwoNodeEncoder(t, 'e', 'r')

	leftTok := se.tokens[i]
	rightTok := se.tokens[j]
	rank, _ := se.tok.GetPairRank(leftTok, rightTok)

	cand := mergeCandidate{
		leftIndex:  i,
		rightIndex: j,
		rank:       rank,
		liveLeft:   se.live[i],
		liveRight:  se.live[j],
	}

	se.liveGen++
	se.live[i] = se.liveGen

	if se.isValidCandidate(cand) {
		t.Fatalf("expected candidate invalid due to live version mismatch")
	}
}

func TestIsValidCandidate_AdjacencyMismatch(t *testing.T) {
	se, i, j := setupTwoNodeEncoder(t, 'e', 'r')

	leftTok := se.tokens[i]
	rightTok := se.tokens[j]
	rank, _ := se.tok.GetPairRank(leftTok, rightTok)

	cand := mergeCandidate{
		leftIndex:  i,
		rightIndex: j,
		rank:       rank,
		liveLeft:   se.live[i],
		liveRight:  se.live[j],
	}

	se.next[i] = -1

	if se.isValidCandidate(cand) {
		t.Fatalf("expected candidate invalid due to broken adjacency")
	}
}

func TestIsValidCandidate_RankMismatch(t *testing.T) {
	se, i, j := setupTwoNodeEncoder(t, 'e', 'r')

	leftTok := se.tokens[i]
	rightTok := se.tokens[j]
	rank, _ := se.tok.GetPairRank(leftTok, rightTok)

	cand := mergeCandidate{
		leftIndex:  i,
		rightIndex: j,
		rank:       rank,
		liveLeft:   se.live[i],
		liveRight:  se.live[j],
	}

	se.tokens[i] = 999999

	if se.isValidCandidate(cand) {
		t.Fatalf("expected candidate invalid due to rank mismatch")
	}
}

func TestIsValidCandidate_DeadNode(t *testing.T) {
	se, i, j := setupTwoNodeEncoder(t, 'e', 'r')

	leftTok := se.tokens[i]
	rightTok := se.tokens[j]
	rank, _ := se.tok.GetPairRank(leftTok, rightTok)

	cand := mergeCandidate{
		leftIndex:  i,
		rightIndex: j,
		rank:       rank,
		liveLeft:   se.live[i],
		liveRight:  se.live[j],
	}

	se.next[i] = -1
	se.prev[j] = -1
	se.next[j] = -1

	if se.isValidCandidate(cand) {
		t.Fatalf("expected invalid due to j being dead")
	}
}

func TestIsValidCandidate_PairNoLongerMergeable(t *testing.T) {
	se, i, j := setupTwoNodeEncoder(t, 'e', 'r')

	leftTok := se.tokens[i]
	rightTok := se.tokens[j]
	rank, _ := se.tok.GetPairRank(leftTok, rightTok)

	cand := mergeCandidate{
		leftIndex:  i,
		rightIndex: j,
		rank:       rank,
		liveLeft:   se.live[i],
		liveRight:  se.live[j],
	}

	se.tokens[i] = se.tok.GetByteToToken(0x00)
	se.tokens[j] = se.tok.GetByteToToken(0xFF)

	_, ok := se.tok.GetPairRank(se.tokens[i], se.tokens[j])
	if ok {
		t.Fatalf("test assumption: 0x00,0xFF must be unmergeable")
	}

	if se.isValidCandidate(cand) {
		t.Fatalf("expected invalid because pair is no longer mergeable")
	}
}

func TestSeedAdjacency_InternalPairs(t *testing.T) {
	tok, err := core.LoadTokenizerFromFiles("../testdata/gpt2/vocab.json", "../testdata/gpt2/merges.txt")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}

	se := NewStreamingEncoderV2(tok)
	mh := &mockHeap{}
	se.heap = mh

	nodes := se.appendBytes([]byte("abc"))
	se.seedAdjacency(nodes)

	if len(mh.items) == 0 {
		t.Fatalf("expected adjacency candidates, got none")
	}

	if got := len(mh.items); got != len(nodes)-1 {
		t.Fatalf("expected %d adjacency pairs, got %d", len(nodes)-1, got)
	}

	for i, c := range mh.items {
		left := nodes[i]
		right := nodes[i+1]

		if c.leftIndex != left || c.rightIndex != right {
			t.Fatalf("expected (%d,%d), got (%d,%d)", left, right, c.leftIndex, c.rightIndex)
		}

		if c.liveLeft != se.live[left] || c.liveRight != se.live[right] {
			t.Fatalf("live version mismatch in candidate %v", c)
		}

		lt := se.tokens[left]
		rt := se.tokens[right]
		rank, ok := tok.GetPairRank(lt, rt)
		if ok && rank != c.rank {
			t.Fatalf("rank mismatch: got %d, expected %d", c.rank, rank)
		}
	}
}

func TestSeedAdjacency_BoundaryAdjacency(t *testing.T) {
	tok, err := core.LoadTokenizerFromFiles("../testdata/gpt2/vocab.json", "../testdata/gpt2/merges.txt")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}

	se := NewStreamingEncoderV2(tok)
	mh := &mockHeap{}
	se.heap = mh

	n1 := se.appendBytes([]byte("a"))
	se.seedAdjacency(n1)
	if len(mh.items) != 0 {
		t.Fatalf("first chunk should not generate boundary candidates")
	}

	n2 := se.appendBytes([]byte("bcd"))
	se.seedAdjacency(n2)

	if len(mh.items) == 0 {
		t.Fatalf("expected boundary adjacency candidate")
	}

	c := mh.items[0]
	expectedLeft := n1[len(n1)-1]
	expectedRight := n2[0]

	if c.leftIndex != expectedLeft || c.rightIndex != expectedRight {
		t.Fatalf("expected boundary (%d,%d), got (%d,%d)",
			expectedLeft, expectedRight, c.leftIndex, c.rightIndex)
	}

	if c.liveLeft != se.live[expectedLeft] || c.liveRight != se.live[expectedRight] {
		t.Fatalf("live versions mismatch at boundary")
	}
}

func TestSeedAdjacency_RankCorrectness(t *testing.T) {
	tok, err := core.LoadTokenizerFromFiles("../testdata/gpt2/vocab.json", "../testdata/gpt2/merges.txt")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}

	se := NewStreamingEncoderV2(tok)
	mh := &mockHeap{}
	se.heap = mh

	nodes := se.appendBytes([]byte("er"))
	se.seedAdjacency(nodes)

	if len(mh.items) != 1 {
		t.Fatalf("expected 1 adjacency candidate, got %d", len(mh.items))
	}

	c := mh.items[0]
	leftTok := se.tokens[nodes[0]]
	rightTok := se.tokens[nodes[1]]
	expectedRank, ok := tok.GetPairRank(leftTok, rightTok)
	if !ok {
		t.Fatalf("test assumption: 'er' must be mergeable in GPT2")
	}

	if c.rank != expectedRank {
		t.Fatalf("candidate rank mismatch: expected %d, got %d", expectedRank, c.rank)
	}
}

func TestSeedAdjacency_UnmergeablePairNoCandidate(t *testing.T) {
	tok, err := core.LoadTokenizerFromFiles("../testdata/gpt2/vocab.json", "../testdata/gpt2/merges.txt")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	se := NewStreamingEncoderV2(tok)

	mh := &mockHeap{}
	se.heap = mh

	nodes := se.appendBytes([]byte{0x00, 0xFF})
	se.seedAdjacency(nodes)

	if len(mh.items) != 0 {
		t.Fatalf("expected no adjacency candidates for unmergeable pair")
	}
}

func TestSeedAdjacency_DoesNotMutateList(t *testing.T) {
	tok, err := core.LoadTokenizerFromFiles("../testdata/gpt2/vocab.json", "../testdata/gpt2/merges.txt")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	se := NewStreamingEncoderV2(tok)
	mh := &mockHeap{}
	se.heap = mh

	nodes := se.appendBytes([]byte("xyz"))

	prevBefore := append([]int(nil), se.prev...)
	nextBefore := append([]int(nil), se.next...)
	tokensBefore := append([]int(nil), se.tokens...)
	liveBefore := append([]uint32(nil), se.live...)

	se.seedAdjacency(nodes)

	if !equalIntSlices(prevBefore, se.prev) ||
		!equalIntSlices(nextBefore, se.next) ||
		!equalIntSlices(tokensBefore, se.tokens) ||
		!equalUintSlices(liveBefore, se.live) {
		t.Fatalf("seedAdjacency mutated core arrays unexpectedly")
	}
}

func encodeStreamingV2(t *testing.T, tok *core.Tokenizer, chunkSizes []int, input []byte) []int {
	enc := NewStreamingEncoderV2(tok)
	out := make([]int, 0, len(input))

	consumed := 0
	for _, size := range chunkSizes {
		if consumed >= len(input) {
			break
		}

		end := consumed + size
		if end > len(input) {
			end = len(input)
		}

		chunk := input[consumed:end]
		consumed = end

		toks := enc.Push(chunk)
		if toks != nil {
			out = append(out, toks...)
		}
	}

	tail := enc.Flush()
	if tail != nil {
		out = append(out, tail...)
	}

	return out
}

func TestAppendBytes_SingleChunk(t *testing.T) {
	tok, err := core.LoadTokenizerFromFiles("../testdata/gpt2/vocab.json", "../testdata/gpt2/merges.txt")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}

	se := NewStreamingEncoderV2(tok)

	chunk := []byte{10, 20, 30}
	nodes := se.appendBytes(chunk)

	if len(nodes) != 3 {
		t.Fatalf("expected 3 nodes, got %d", len(nodes))
	}

	if se.head != nodes[0] {
		t.Fatalf("expected head=%d got %d", nodes[0], se.head)
	}
	if se.tail != nodes[2] {
		t.Fatalf("expected tail=%d got %d", nodes[2], se.tail)
	}

	for i, idx := range nodes {
		expected := tok.GetByteToToken(chunk[i])
		if se.tokens[idx] != expected {
			t.Fatalf("tokens[%d] = %d, expected %d", idx, se.tokens[idx], expected)
		}
	}

	if se.prev[nodes[0]] != -1 {
		t.Fatalf("prev of first node must be -1")
	}
	if se.next[nodes[2]] != -1 {
		t.Fatalf("next of last node must be -1")
	}

	if se.next[nodes[0]] != nodes[1] {
		t.Fatalf("next mismatch: %d -> expected %d", se.next[nodes[0]], nodes[1])
	}
	if se.prev[nodes[1]] != nodes[0] {
		t.Fatalf("prev mismatch: %d -> expected %d", se.prev[nodes[1]], nodes[0])
	}
}

func TestAppendBytes_MultipleChunks(t *testing.T) {
	tok, err := core.LoadTokenizerFromFiles("../testdata/gpt2/vocab.json", "../testdata/gpt2/merges.txt")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}

	se := NewStreamingEncoderV2(tok)

	chunk1 := []byte{1, 2, 3}
	chunk2 := []byte{4, 5}

	n1 := se.appendBytes(chunk1)
	n2 := se.appendBytes(chunk2)

	if se.head != n1[0] {
		t.Fatalf("head should remain first chunk start")
	}
	if se.tail != n2[len(n2)-1] {
		t.Fatalf("tail should point to last node of second chunk")
	}

	if se.next[n1[len(n1)-1]] != n2[0] {
		t.Fatalf("expected next of last node of chunk1 to be %d", n2[0])
	}
	if se.prev[n2[0]] != n1[len(n1)-1] {
		t.Fatalf("expected prev of first node of chunk2 to be %d", n1[len(n1)-1])
	}
}

func TestAppendBytes_LiveVersions(t *testing.T) {
	tok, err := core.LoadTokenizerFromFiles("../testdata/gpt2/vocab.json", "../testdata/gpt2/merges.txt")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	se := NewStreamingEncoderV2(tok)

	se.appendBytes([]byte{9, 9})

	se.appendBytes([]byte{9})
	lv2 := append([]uint32{}, se.live...)

	for i := 1; i < len(lv2); i++ {
		if lv2[i] <= lv2[i-1] {
			t.Fatalf("live version did not increase strictly")
		}
	}

	if !(lv2[0] < lv2[1] && lv2[1] < lv2[2]) {
		t.Fatalf("live versions not strictly monotonic: %v", lv2)
	}
}

func TestAppendBytes_LongSequence(t *testing.T) {
	tok, err := core.LoadTokenizerFromFiles("../testdata/gpt2/vocab.json", "../testdata/gpt2/merges.txt")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	se := NewStreamingEncoderV2(tok)

	total := 0
	for i := 0; i < 50; i++ {
		chunk := []byte{byte(i), byte(i + 1)}
		se.appendBytes(chunk)
		total += 2
	}

	count := 0
	cur := se.head

	for cur != -1 {
		count++
		if cur == se.tail {
			break
		}
		cur = se.next[cur]
	}

	if count != total {
		t.Fatalf("expected %d nodes in list, found %d", total, count)
	}
}

func TestStreamingE2E_SimpleTwoChunk(t *testing.T) {
	tok, err := core.LoadTokenizerFromFiles("../testdata/gpt2/vocab.json", "../testdata/gpt2/merges.txt")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}

	se := NewStreamingEncoderV2(tok)

	full := "hello world"
	c1 := "hello "
	c2 := "world"

	out := []int{}
	out = append(out, se.Push([]byte(c1))...)
	out = append(out, se.Push([]byte(c2))...)
	out = append(out, se.Flush()...)

	want := tok.EncodeOffline([]byte(full), nil)

	if !reflect.DeepEqual(out, want) {
		t.Fatalf("streaming mismatch:\ngot  %v\nwant %v", out, want)
	}
}

func TestStreamingE2E_MultiChunk_NoCrossBoundaryMerges(t *testing.T) {
	tok, err := core.LoadTokenizerFromFiles("../testdata/gpt2/vocab.json", "../testdata/gpt2/merges.txt")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}

	se := NewStreamingEncoderV2(tok)

	full := "XYZ123"
	c1 := "XYZ"
	c2 := "123"

	out := []int{}
	out = append(out, se.Push([]byte(c1))...)
	out = append(out, se.Push([]byte(c2))...)
	out = append(out, se.Flush()...)

	want := tok.EncodeOffline([]byte(full), nil)

	if !reflect.DeepEqual(out, want) {
		t.Fatalf("streaming mismatch:\ngot  %v\nwant %v", out, want)
	}
}

func TestStreamingE2E_MultiChunk_WithCrossBoundaryMerges(t *testing.T) {
	tok, err := core.LoadTokenizerFromFiles("../testdata/gpt2/vocab.json", "../testdata/gpt2/merges.txt")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}

	se := NewStreamingEncoderV2(tok)

	full := "hello world"

	chunks := []string{
		"hell",
		"o ",
		"wo",
		"rld",
	}

	out := []int{}
	for _, c := range chunks {
		out = append(out, se.Push([]byte(c))...)
	}
	out = append(out, se.Flush()...)

	want := tok.EncodeOffline([]byte(full), nil)

	if !reflect.DeepEqual(out, want) {
		t.Fatalf("streaming mismatch:\ngot  %v\nwant %v", out, want)
	}
}

func TestStreamingE2E_ByteByByte(t *testing.T) {
	tok, err := core.LoadTokenizerFromFiles("../testdata/gpt2/vocab.json", "../testdata/gpt2/merges.txt")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	se := NewStreamingEncoderV2(tok)

	input := "hello world"

	out := []int{}
	for i := 0; i < len(input); i++ {
		out = append(out, se.Push([]byte{input[i]})...)
	}
	out = append(out, se.Flush()...)

	want := tok.EncodeOffline([]byte(input), nil)

	if !reflect.DeepEqual(out, want) {
		t.Fatalf("byte-by-byte mismatch:\ngot  %v\nwant %v", out, want)
	}
}

func TestStreamingE2E_RandomChunkSplits(t *testing.T) {
	tok, err := core.LoadTokenizerFromFiles("../testdata/gpt2/vocab.json", "../testdata/gpt2/merges.txt")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	se := NewStreamingEncoderV2(tok)

	input := "The quick brown fox jumps over the lazy dog"

	splits := []int{1, 3, 2, 5, 4, 1, 6, 3, 10, 2, 8, 1}

	out := []int{}
	i := 0
	for _, n := range splits {
		if i >= len(input) {
			break
		}
		end := i + n
		if end > len(input) {
			end = len(input)
		}
		out = append(out, se.Push([]byte(input[i:end]))...)
		i = end
	}
	out = append(out, se.Flush()...)

	want := tok.EncodeOffline([]byte(input), nil)

	if !reflect.DeepEqual(out, want) {
		t.Fatalf("random-chunk mismatch:\ngot  %v\nwant %v", out, want)
	}
}

func TestStreamingE2E_SpaceMerges(t *testing.T) {
	tok, err := core.LoadTokenizerFromFiles("../testdata/gpt2/vocab.json", "../testdata/gpt2/merges.txt")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	se := NewStreamingEncoderV2(tok)

	out := []int{}
	out = append(out, se.Push([]byte("The"))...)
	out = append(out, se.Push([]byte(" qu"))...)
	out = append(out, se.Push([]byte("ick"))...)
	out = append(out, se.Push([]byte(" brown"))...)
	out = append(out, se.Push([]byte(" fox"))...)
	out = append(out, se.Flush()...)

	want := tok.EncodeOffline([]byte("The quick brown fox"), nil)

	if !reflect.DeepEqual(out, want) {
		t.Fatalf("space-merge mismatch:\ngot  %v\nwant %v", out, want)
	}
}
func TestStreamingE2E_UnicodeBoundary(t *testing.T) {
	tok, err := core.LoadTokenizerFromFiles("../testdata/gpt2/vocab.json", "../testdata/gpt2/merges.txt")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	se := NewStreamingEncoderV2(tok)

	input := "Héllo 🌍"

	out := []int{}
	out = append(out, se.Push([]byte("Hé"))...)
	out = append(out, se.Push([]byte("ll"))...)
	out = append(out, se.Push([]byte("o "))...)
	out = append(out, se.Push([]byte("🌍"))...)
	out = append(out, se.Flush()...)

	want := tok.EncodeOffline([]byte(input), nil)

	if !reflect.DeepEqual(out, want) {
		t.Fatalf("unicode-boundary mismatch:\ngot  %v\nwant %v", out, want)
	}
}

func TestStreamingE2E_FlushOnly(t *testing.T) {
	tok, err := core.LoadTokenizerFromFiles("../testdata/gpt2/vocab.json", "../testdata/gpt2/merges.txt")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	se := NewStreamingEncoderV2(tok)

	se.Push([]byte("hello "))
	got := se.Flush()
	want := tok.EncodeOffline([]byte("hello "), nil)

	if !reflect.DeepEqual(got, want) {
		t.Fatalf("flush mismatch:\ngot %v\nwant %v", got, want)
	}
}

func TestStreamingE2E_TailReserveInvariant(t *testing.T) {
	tok, err := core.LoadTokenizerFromFiles("../testdata/gpt2/vocab.json", "../testdata/gpt2/merges.txt")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	se := NewStreamingEncoderV2(tok)

	input := "hello world!!!"

	// Push in tiny pieces to force long live tail
	out := []int{}
	out = append(out, se.Push([]byte("hel"))...)
	out = append(out, se.Push([]byte("lo"))...)
	out = append(out, se.Push([]byte(" w"))...)
	out = append(out, se.Push([]byte("or"))...)
	out = append(out, se.Push([]byte("ld"))...)
	out = append(out, se.Push([]byte("!!"))...)
	out = append(out, se.Push([]byte("!"))...)
	out = append(out, se.Flush()...)

	want := tok.EncodeOffline([]byte(input), nil)

	if !reflect.DeepEqual(out, want) {
		t.Fatalf("tail-reserve mismatch:\ngot  %v\nwant %v", out, want)
	}
}

func TestStreaming_ListInvariants(t *testing.T) {
	tok, err := core.LoadTokenizerFromFiles("../testdata/gpt2/vocab.json", "../testdata/gpt2/merges.txt")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	se := NewStreamingEncoderV2(tok)

	input := "hello world this is a long-ish string for testing"
	se.Push([]byte(input))

	// Walk forward
	forward := []int{}
	seen := map[int]bool{}

	idx := se.head
	for idx != -1 {
		if idx < 0 || idx >= len(se.tokens) {
			t.Fatalf("out-of-bounds next pointer: %d", idx)
		}
		if seen[idx] {
			t.Fatalf("cycle detected at %d", idx)
		}
		seen[idx] = true

		forward = append(forward, idx)
		idx = se.next[idx]
	}

	if len(forward) == 0 {
		t.Fatalf("empty forward traversal")
	}
	if se.prev[se.head] != -1 {
		t.Fatalf("head.prev != -1")
	}
	if se.tail != forward[len(forward)-1] {
		t.Fatalf("tail mismatch: got %d want %d", se.tail, forward[len(forward)-1])
	}
	if se.next[se.tail] != -1 {
		t.Fatalf("tail.next != -1")
	}

	// Walk backward
	backward := []int{}
	idx = se.tail
	for idx != -1 {
		backward = append(backward, idx)
		idx = se.prev[idx]
	}

	// backward reversed must equal forward
	for i := range forward {
		if forward[i] != backward[len(backward)-1-i] {
			t.Fatalf("bidirectional mismatch: forward=%v backward=%v", forward, backward)
		}
	}
}

func TestStreaming_FrontierConsistency(t *testing.T) {
	tok, err := core.LoadTokenizerFromFiles("../testdata/gpt2/vocab.json", "../testdata/gpt2/merges.txt")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	se := NewStreamingEncoderV2(tok)

	// Create a spy heap that tracks all Push calls
	seen := map[[2]int]bool{}
	originalHeap := se.heap

	spyHeap := &spyHeapForFrontierTest{
		wrapped: originalHeap,
		onPush: func(c mergeCandidate) {
			seen[[2]int{c.leftIndex, c.rightIndex}] = true
		},
	}
	se.heap = spyHeap

	// String chosen intentionally with dense merges: "aaaaa"
	input := []byte("aaaaa")

	se.Push(input)
	se.Flush()

	// Validate that:
	//   For each merge that happened, only its two neighbors were added.
	// This is indirect: we ensure no impossible (i,j) pairs appeared.
	for pair := range seen {
		i, j := pair[0], pair[1]
		if i < 0 || j < 0 {
			t.Fatalf("invalid pair pushed: %+v", pair)
		}
		if j != se.next[i] && i != se.prev[j] {
			t.Fatalf("non-adjacent pair was added to heap: %v", pair)
		}
	}
}

func TestStreaming_FuzzOfflineEquivalence(t *testing.T) {
	tok, err := core.LoadTokenizerFromFiles("../testdata/gpt2/vocab.json", "../testdata/gpt2/merges.txt")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}

	for iter := 0; iter < 250; iter++ {
		// Random UTF-8 text
		n := 50 + rand.Intn(200)
		runes := make([]rune, n)
		for i := range runes {
			runes[i] = rune(32 + rand.Intn(2000))
		}
		s := string(runes)

		se := NewStreamingEncoderV2(tok)

		// Push byte-by-byte to maximize stress
		var out []int
		for i := 0; i < len(s); i++ {
			out = append(out, se.Push([]byte{s[i]})...)
		}
		out = append(out, se.Flush()...)

		want := tok.EncodeOffline([]byte(s), nil)

		if !reflect.DeepEqual(out, want) {
			t.Fatalf("fuzz mismatch:\ninput=%q\ngot  %v\nwant %v", s, out, want)
		}
	}
}

func TestStreaming_CrossBoundaryFuzzer(t *testing.T) {
	tok, err := core.LoadTokenizerFromFiles("../testdata/gpt2/vocab.json", "../testdata/gpt2/merges.txt")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}

	for iter := 0; iter < 100; iter++ {
		// random input
		n := 80 + rand.Intn(120)
		data := make([]byte, n)
		for i := range data {
			data[i] = byte(32 + rand.Intn(90))
		}

		se := NewStreamingEncoderV2(tok)

		// apply random chunking
		out := []int{}
		i := 0
		for i < len(data) {
			sz := 1 + rand.Intn(10)
			end := i + sz
			if end > len(data) {
				end = len(data)
			}
			out = append(out, se.Push(data[i:end])...)
			i = end
		}
		out = append(out, se.Flush()...)

		want := tok.EncodeOffline(data, nil)

		if !reflect.DeepEqual(out, want) {
			t.Fatalf("cross-boundary mismatch:\ninput=%q\ngot  %v\nwant %v", string(data), out, want)
		}
	}
}

func TestStreaming_TailReserveSweep(t *testing.T) {
	tok, err := core.LoadTokenizerFromFiles("../testdata/gpt2/vocab.json", "../testdata/gpt2/merges.txt")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}

	input := []byte("The quick brown fox jumped over the log while thinking about tokens")

	for tailRes := 0; tailRes <= 10; tailRes++ {
		se := NewStreamingEncoderV2(tok)
		se.tailReserve = tailRes

		// random push pattern
		out := []int{}
		i := 0
		for i < len(input) {
			n := 1 + rand.Intn(4)
			end := i + n
			if end > len(input) {
				end = len(input)
			}
			out = append(out, se.Push(input[i:end])...)
			i = end
		}

		// final flush
		out = append(out, se.Flush()...)

		// Validate tail invariants via flush output; matches offline
		want := tok.EncodeOffline(input, nil)
		if !reflect.DeepEqual(out, want) {
			t.Fatalf("tailReserve=%d mismatch:\ngot  %v\nwant %v", tailRes, out, want)
		}
	}
}

func TestStreaming_Stress1MB(t *testing.T) {
	tok, err := core.LoadTokenizerFromFiles("../testdata/gpt2/vocab.json", "../testdata/gpt2/merges.txt")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	se := NewStreamingEncoderV2(tok)

	// 1 MB of mostly-printable data
	data := make([]byte, 1<<20)
	for i := range data {
		data[i] = byte(32 + rand.Intn(90))
	}

	out := []int{}
	const chunk = 4096
	for i := 0; i < len(data); i += chunk {
		end := i + chunk
		if end > len(data) {
			end = len(data)
		}
		out = append(out, se.Push(data[i:end])...)
	}
	out = append(out, se.Flush()...)

	want := tok.EncodeOffline(data, nil)

	if !reflect.DeepEqual(out, want) {
		t.Fatalf("1MB stress mismatch: got=%d want=%d tokens", len(out), len(want))
	}
}

func intSliceToBytes(xs []int) []byte {
	b := make([]byte, len(xs)*4)
	for i, v := range xs {
		u := uint32(v)
		b[i*4+0] = byte(u >> 24)
		b[i*4+1] = byte(u >> 16)
		b[i*4+2] = byte(u >> 8)
		b[i*4+3] = byte(u)
	}
	return b
}

func equalIntSlices(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func equalUintSlices(a, b []uint32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func newSyntheticList(se *StreamingEncoderV2, lens []int) (indices []int) {
	n := len(lens)
	indices = make([]int, n)

	startIndex := len(se.tokens)

	se.tokens = append(se.tokens, make([]int, n)...)
	se.prev = append(se.prev, make([]int, n)...)
	se.next = append(se.next, make([]int, n)...)
	se.live = append(se.live, make([]uint32, n)...)

	for i := 0; i < n; i++ {
		idx := startIndex + i
		indices[i] = idx

		se.tokens[idx] = i

		if se.syntheticLengths == nil {
			se.syntheticLengths = make(map[int]int)
		}
		se.syntheticLengths[i] = lens[i]

		if i == 0 {
			se.prev[idx] = -1
		} else {
			se.prev[idx] = startIndex + i - 1
		}
		if i == n-1 {
			se.next[idx] = -1
		} else {
			se.next[idx] = startIndex + i + 1
		}

		se.liveGen++
		se.live[idx] = se.liveGen
	}

	se.head = indices[0]
	se.tail = indices[n-1]

	return indices
}
