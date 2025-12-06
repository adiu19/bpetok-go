package streaming_encoder_naive

import "github.com/bpetok/internal/tokenizer/core"

// NaiveStreamingEncoderState implements a NAIVE (greedy) streaming encoder by buffering input bytes
// and greedily flushing any prefix that is guaranteed not to participate in
// future merges (based off of the max token size in our vocab). The final lMax-1 bytes are held back as a safety margin so
// merges that span chunk boundaries are preserved.
type NaiveStreamingEncoderState struct {
	tok         *core.Tokenizer
	tailReserve int

	buf    []byte
	outBuf []int

	// optimization flags
	optPreAllocScratch bool
	optFlattenLookup   bool
	optHotLoopTighten  bool
	optOutBufReuse     bool
	optNoCopyReturn    bool
}

// NewNaiveStreamingEncoderState returns a new instance of the encoder state with opt params disabled.
func NewNaiveStreamingEncoderState(t *core.Tokenizer) *NaiveStreamingEncoderState {
	tail := 0
	if t.MaxTokenByteLen > 0 {
		tail = t.MaxTokenByteLen - 1
	}

	return &NaiveStreamingEncoderState{
		tok:                t,
		tailReserve:        tail,
		optPreAllocScratch: false,
		optFlattenLookup:   false,
		optHotLoopTighten:  false,
		optOutBufReuse:     true,
		optNoCopyReturn:    true,
	}
}

// NewNaiveStreamingEncoderStateWithOpts returns a new instance of the encoder state with opt params.
func NewNaiveStreamingEncoderStateWithOpts(t *core.Tokenizer, optPreAllocScratch bool, optFlattenLookup bool, optHotLoopTighten bool, optOutBufReuse bool, optNoCopyReturn bool) *NaiveStreamingEncoderState {
	tail := 0
	if t.MaxTokenByteLen > 0 {
		tail = t.MaxTokenByteLen - 1
	}

	st := NaiveStreamingEncoderState{
		tok:                t,
		tailReserve:        tail,
		optPreAllocScratch: optPreAllocScratch,
		optFlattenLookup:   optFlattenLookup,
		optHotLoopTighten:  optHotLoopTighten,
		optOutBufReuse:     optOutBufReuse,
		optNoCopyReturn:    optNoCopyReturn,
	}

	if st.optOutBufReuse {
		st.outBuf = make([]int, 64*1024) // 64KB pre-allocated outbut buffer
	}

	return &st
}

// returnOut returns the output buffer - either copied or via a reference pointer depending on our flags
func (st *NaiveStreamingEncoderState) returnOut() []int {
	if st.optNoCopyReturn {
		return st.outBuf
	}

	out := make([]int, len(st.outBuf))
	copy(out, st.outBuf)
	return out
}

// Push consumes the next chunk of raw bytes and emits any finalized tokens.
func (st *NaiveStreamingEncoderState) Push(chunk []byte) []int {
	st.outBuf = st.outBuf[:0]
	if len(chunk) > 0 {
		st.buf = append(st.buf, chunk...)
	}

	st.emitCommitted()

	if len(st.outBuf) == 0 {
		return nil
	}
	return st.returnOut()
}

// Flush encodes whatever bytes remain in the internal buffer.
func (st *NaiveStreamingEncoderState) Flush() []int {
	st.outBuf = st.outBuf[:0]
	if len(st.buf) > 0 {
		tokens := st.tok.EncodeOffline(st.buf)
		st.outBuf = append(st.outBuf, tokens...)
		st.buf = st.buf[:0]
	}

	if len(st.outBuf) == 0 {
		return nil
	}

	return st.returnOut()
}

func (st *NaiveStreamingEncoderState) emitCommitted() {
	emitLimit := len(st.buf) - st.tailReserve
	if emitLimit <= 0 {
		return
	}

	tokens := st.tok.EncodeOffline(st.buf)

	consumed := 0
	for _, id := range tokens {
		tokLen := st.tok.TokenLen(id)
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
