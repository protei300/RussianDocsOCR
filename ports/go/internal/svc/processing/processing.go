// Package processing decides which processing mode is in force: documents stored, or recognised
// in memory only.
//
// PROCESSING_MODE=stored (the default) is the service as it always was. PROCESSING_MODE=memory is
// for deployments that must not keep personal data at all: the document routes, the background
// worker and the seeding are switched off, and POST /recognize is the only way a document gets in
// — decoded, recognised and answered inside the request, with nothing written.
//
// **Resolution never fails, and when the value is wrong it fails toward privacy.** That is the
// opposite of AUTH_MODE, which falls back to the old behaviour, and the difference is deliberate:
// somebody who typed PROCESSING_MODE=memroy asked for documents not to be stored, and quietly
// storing them would break the one promise the setting exists to make.
//
//	unset, empty   -> stored   (backward compatible: the default is not a mistake)
//	"stored"       -> stored
//	"memory"       -> memory
//	anything else  -> memory, with a reason that is logged and shown on /status
//
// Configuration only, never a runtime setting: a switch on a web page would let anyone with admin
// rights turn storage back on for a service that promised not to store.
//
// Port of service/core/processing.py; the contract is ports/MEMORY.md §1.
package processing

import (
	"fmt"
	"strings"

	"github.com/protei300/RussianDocsOCR/ports/go/internal/svc/auth"
)

// The two modes. Strings, because they go on the wire in /auth/config and /status.
const (
	Stored = "stored"
	Memory = "memory"
)

// Resolve returns (mode, reason); reason is nil when the configured value was used. The ONE
// function that reads the raw PROCESSING_MODE.
func Resolve(raw string) (string, *string) {
	value := strings.ToLower(strings.TrimSpace(raw))
	switch value {
	case "":
		return Stored, nil
	case Stored, Memory:
		return value, nil
	}
	reason := fmt.Sprintf("PROCESSING_MODE=%s is not one of %s, %s — using 'memory' (documents "+
		"are not stored), the safe reading of a mistyped privacy setting",
		auth.PyRepr(value), Stored, Memory)
	return Memory, &reason
}
