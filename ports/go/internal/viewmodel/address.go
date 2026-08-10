package viewmodel

// The address block: present only for INTPASSPORTADDR, and NOT PRODUCED by this port yet.
//
// The types are declared even though nothing fills them, deliberately. The contract in
// spec/viewmodel.md includes them, an integrator reading this package needs to see the
// shape, and an omitted type reads as an oversight that the next port "helpfully" invents
// differently. `address` is therefore always null here, which is also its value for every
// other document type.
//
// What is missing before this can be built: the OBB detector and the printed-versus-
// handwritten classifier (both deferred with INTPASSPORTADDR), and an ANONYMISED sample.
// No such sample exists in the repository, so the path has no golden and cannot be graded
// — which is why it is deferred rather than written blind.

// Address is present only for INTPASSPORTADDR.
type Address struct {
	// Aligned is false when the geometry and text lists desynchronised. A consumer must
	// then suppress the overlay rather than caption boxes with the wrong text.
	Aligned bool          `json:"aligned"`
	Lines   []AddressLine `json:"lines"`
}

type AddressLine struct {
	ID           string   `json:"id"`
	Kind         *string  `json:"kind"`
	Text         *string  `json:"text"`
	PHandwritten *float64 `json:"p_handwritten"`
	Obbox        *Obbox   `json:"obbox"`
}

type Obbox struct {
	Cx *float64 `json:"cx"`
	Cy *float64 `json:"cy"`
	W  *float64 `json:"w"`
	H  *float64 `json:"h"`
	// AngleRad is RADIANS at six decimals, not the wire default of four.
	AngleRad *float64 `json:"angle_rad"`
	Conf     *float64 `json:"conf"`
	Label    *string  `json:"label"`
}
