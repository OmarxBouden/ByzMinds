package ledger

import "time"

func defaultCommitTime() uint64 { return uint64(time.Now().UnixNano()) }
