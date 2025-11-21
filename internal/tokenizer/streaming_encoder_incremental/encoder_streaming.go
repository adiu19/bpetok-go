package tokenizer

type EncoderStreaming struct {
	tok *Tokenizer
	

	 // persistent linked list state
	 tokens []int
	 prev   []int
	 next   []int
	 live   []int

	 // heap of merge candidates (incremental)
	 heap *utils.MergeHeap

	  // pointer to the current tail index
	  head int
	  tail int
}

func NewEncoderStreaming(tok *Tokenizer) *EncoderStreaming {
	return &EncoderStreaming{
		tok: tok,
	}
}