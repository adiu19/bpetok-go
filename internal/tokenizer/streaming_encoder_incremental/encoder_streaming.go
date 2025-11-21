package streaming_encoder_incremental

type EncoderStreaming struct {
	tok *Tokenizer

	// persistent linked list state
	tokens  []int // token IDs
	prev    []int
	next    []int
	live    []uint32 // version per node
	liveGen uint32   // monotonically increasing global counter

	// heap of merge candidates (incremental)
	heap *utils.MergeHeap

	// pointer to the current tail index
	head int
	tail int

	outBuf []int // output buffer for tokens
}

// NewStreamingEncoderV2 creates a new incremental encoder instance.
func NewStreamingEncoderV2(tok *core.Tokenizer) *StreamingEncoderV2 {
    return &StreamingEncoderV2{
        tok: tok,
        head: -1,
        tail: -1,
        liveGen: 1,
        outBuf: make([]int, 0, 128),   // arbitrary initial cap
	}
}

// Push ingests the next chunk of raw bytes, performs incremental merges, and returns any committed tokens.
func (se *StreamingEncoderV2) Push(chunk []byte) []int {
    // TODO: implement incremental append + merge frontier update + commit
    return nil
}


// Flush finalizes all remaining tokens.
func (se *StreamingEncoderV2) Flush() []int {
    // TODO: commit all remaining live tokens
    return nil
}