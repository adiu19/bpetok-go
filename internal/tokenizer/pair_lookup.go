package tokenizer

// PairLookup provides fast lookup of pair info (rank and token) using a hybrid approach:
// - 2D array for pairs where both tokens are < fastLookupSize (O(1) lookup)
// - Map fallback for larger pairs
type PairLookup struct {
	fastLookup     [][]uint64
	fastLookupSize int
	fallback       map[uint64]uint64
}

// NewPairLookup creates a new pair lookup structure
func NewPairLookup(pairInfo map[uint64]uint64, vocabSize int) *PairLookup {
	fastLookupSize := 2048
	if vocabSize < fastLookupSize {
		fastLookupSize = vocabSize
	}

	fastLookup := make([][]uint64, fastLookupSize)
	for i := range fastLookup {
		fastLookup[i] = make([]uint64, fastLookupSize)
		for j := range fastLookup[i] {
			fastLookup[i][j] = ^uint64(0)
		}
	}

	fallback := make(map[uint64]uint64, len(pairInfo)/10)

	for key, value := range pairInfo {
		a := int(key >> 32)
		b := int(key & 0xFFFFFFFF)

		if a < fastLookupSize && b < fastLookupSize {
			fastLookup[a][b] = value
		} else {
			fallback[key] = value
		}
	}

	return &PairLookup{
		fastLookup:     fastLookup,
		fastLookupSize: fastLookupSize,
		fallback:       fallback,
	}
}

// Lookup returns the pair info (rank << 32 | tokenID) and whether it was found
func (pl *PairLookup) Lookup(a, b int) (uint64, bool) {
	if a >= 0 && a < pl.fastLookupSize && b >= 0 && b < pl.fastLookupSize {
		value := pl.fastLookup[a][b]
		if value+1 != 0 {
			return value, true
		}
		return 0, false
	}

	key := packPair(a, b)
	value, ok := pl.fallback[key]
	return value, ok
}
