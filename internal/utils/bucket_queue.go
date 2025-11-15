package utils


type BucketQueue struct {
	buckets    [][]MergeCand
	current    int           
	totalCount int           
}

func NewBucketQueue(maxRank int) *BucketQueue {
	return &BucketQueue{
		buckets: make([][]MergeCand, maxRank+1),
		current: 0,
	}
}

func (bq *BucketQueue) Len() int {
	return bq.totalCount
}

func (bq *BucketQueue) Push(c MergeCand) {
	rank := c.Rank
	if rank >= len(bq.buckets) {
		newBuckets := make([][]MergeCand, rank+1)
		copy(newBuckets, bq.buckets)
		bq.buckets = newBuckets
	}
	
	bucket := bq.buckets[rank]
	bucketLen := len(bucket)
	
	var insertPos int
	if bucketLen < 16 {
		insertPos = bucketLen
		for i := 0; i < bucketLen; i++ {
			if bucket[i].Pos >= c.Pos {
				insertPos = i
				break
			}
		}
	} else {
		left, right := 0, bucketLen
		for left < right {
			mid := (left + right) / 2
			if bucket[mid].Pos < c.Pos {
				left = mid + 1
			} else {
				right = mid
			}
		}
		insertPos = left
	}
	
	if insertPos == bucketLen {
		bucket = append(bucket, c)
	} else {
		bucket = append(bucket, MergeCand{})
		copy(bucket[insertPos+1:], bucket[insertPos:])
		bucket[insertPos] = c
	}
	bq.buckets[rank] = bucket
	bq.totalCount++
}

func (bq *BucketQueue) Pop() (MergeCand, bool) {
	for bq.current < len(bq.buckets) && len(bq.buckets[bq.current]) == 0 {
		bq.current++
	}
	
	if bq.current >= len(bq.buckets) {
		return MergeCand{}, false
	}
	
	bucket := bq.buckets[bq.current]
	c := bucket[0]
	bq.buckets[bq.current] = bucket[1:]
	bq.totalCount--
	
	return c, true
}

