package tokenizer

// EncoderState implements a simple streaming encoder by buffering input bytes
// and greedily flushing any prefix that is guaranteed not to participate in
// future merges. The final lMax-1 bytes are held back as a safety margin so
// merges that span chunk boundaries are preserved.
type EncoderState struct {
	tok         *Tokenizer
	tailReserve int

	buf    []byte
	outBuf []int
}

// NewEncoderState returns a new instance of the encoder state.
func NewEncoderState(t *Tokenizer) *EncoderState {
	tail := 0
	if t.MaxTokenByteLen > 0 {
		tail = t.MaxTokenByteLen - 1
	}

	return &EncoderState{
		tok:         t,
		tailReserve: tail,
	}
}

// Push consumes the next chunk of raw bytes and emits any finalized tokens.
func (st *EncoderState) Push(chunk []byte) []int {
	st.outBuf = st.outBuf[:0]
	if len(chunk) > 0 {
		st.buf = append(st.buf, chunk...)
	}

	st.emitCommitted()

	if len(st.outBuf) == 0 {
		return nil
	}
	return append([]int(nil), st.outBuf...)
}

// Flush encodes whatever bytes remain in the internal buffer.
func (st *EncoderState) Flush() []int {
	st.outBuf = st.outBuf[:0]
	if len(st.buf) > 0 {
		tokens := st.tok.EncodeOffline(st.buf)
		st.outBuf = append(st.outBuf, tokens...)
		st.buf = st.buf[:0]
	}

	if len(st.outBuf) == 0 {
		return nil
	}
	return append([]int(nil), st.outBuf...)
}

func (st *EncoderState) emitCommitted() {
	emitLimit := len(st.buf) - st.tailReserve
	if emitLimit <= 0 {
		return
	}

	tokens := st.tok.EncodeOffline(st.buf)

	consumed := 0
	for _, id := range tokens {
		tokLen := len(st.tok.revVocab[id])
		if consumed+tokLen > emitLimit {
			break
		}

		st.outBuf = append(st.outBuf, id)
		consumed += tokLen
	}

	if consumed > 0 {
		st.buf = st.buf[consumed:]
	}
}
