package bpetok

// Encoder interface
type Encoder interface {
	/*
		Feed consumes the next chunk of raw bytes from the input stream. It may emit zero or more
		completed token IDs.
		The returned slice is allowed to alias internal memory (zero-copy) so the caller must treat it as read-only
		and make a copy if they want edits.
	*/
	Feed(chunk []byte) []int

	/*
		Flush tells the encoder that the stream is complete. It returns any remaining token IDs that were buffered
		because they were being waited on to see if there are more merges to apply on them. After flush, the encoder
		is reset to a clean state and can be reused for a new stream.
	*/
	Flush() []int
}

// Decoder interface, no need for flush right now because we won't be maintaining internal buffer
type Decoder interface {
	/*
		Feed consumes token IDs and returns zero or more decoded bytes. Same as the encoder, there is a zero-copy rule;
		returned slice can alias internal memory, call must treat it as read-only
	*/
	Feed(tokens []int) []byte
}

// Tokenizer model
type Tokenizer struct {
}

// LoadTokenizer load the internal tables
func LoadTokenizer(vocab []byte, merges []byte) (*Tokenizer, error)

// NewEncoder
func (t *Tokenizer) NewEncoder() Encoder

// NewDocoder
func (t *Tokenizer) NewDocoder() Decoder
