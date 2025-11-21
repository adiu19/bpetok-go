package core

// Decode a given sequence of tokens to a sequence of bytes
func (t *Tokenizer) Decode(tokens []int) []byte {
	if len(tokens) == 0 {
		return nil
	}

	total := 0
	for _, id := range tokens {
		if id < 0 || id >= len(t.RevVocab) {
			panic("token id out of range while decoding")
		}

		total += len(t.RevVocab[id])
	}

	out := make([]byte, 0, total)
	for _, id := range tokens {
		out = append(out, t.RevVocab[id]...)
	}

	return out
}
