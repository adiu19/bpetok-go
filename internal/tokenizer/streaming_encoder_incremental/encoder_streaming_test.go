package streaming_encoder_incremental

import (
	"reflect"
	"testing"

	"github.com/bpetok/internal/tokenizer/core"
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

	want := tok.EncodeOffline([]byte(full))

	if !reflect.DeepEqual(out, want) {
		t.Fatalf("streaming mismatch:\ngot  %v\nwant %v", out, want)
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
