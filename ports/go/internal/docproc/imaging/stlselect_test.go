package imaging

import "testing"

// lcgResponses reproduces the generator the expected vectors were made with.
func lcgResponses(n, mod int) []float64 {
	x := uint64(12345)
	out := make([]float64, n)
	for i := range out {
		x = (x*1103515245 + 12345) % (1 << 31)
		out[i] = float64((x >> 16) % uint64(mod))
	}
	return out
}

// The expected orders come from a step-for-step model of MSVC's std::nth_element /
// std::partition that reproduces the reference's own SIFT keypoint list element for element
// (8779 raw keypoints cut to 6000, STS_2019 template, 2026-10-08). Responses are quantised so
// that ties, the fat-pivot paths and the boundary partition are all exercised.
func TestRetainBestOrderMatchesMsvc(t *testing.T) {
	cases := []struct {
		n, mod, keep int
		want         []int
	}{
		{300, 50, 120, []int{37, 1, 110, 3, 205, 5, 6, 202, 201, 193, 115, 190, 117, 13, 189, 119, 16, 120, 188, 19, 20, 186, 183, 23, 24, 25, 124, 27, 182, 126, 30, 127, 128, 172, 34, 35, 169, 225, 168, 153, 40, 41, 42, 43, 44, 133, 134, 47, 48, 135, 136, 167, 138, 148, 54, 55, 140, 164, 163, 59, 159, 152, 297, 63, 64, 293, 292, 283, 282, 274, 270, 267, 265, 264, 263, 261, 76, 256, 255, 79, 80, 81, 253, 83, 84, 252, 155, 87, 246, 89, 241, 91, 234, 93, 232, 95, 154, 262, 224, 221, 100, 217, 215, 214, 213, 105, 106, 212, 108, 109, 144, 74, 158, 179, 230, 57, 288, 294, 298, 86, 96, 132, 139}},
		{45, 10, 20, []int{0, 1, 2, 3, 4, 44, 39, 32, 25, 19, 38, 11, 12, 13, 23, 15, 14, 21, 5, 42}},
		{20, 7, 5, []int{0, 1, 3, 7, 2, 9, 10, 16}},
		{500, 1000, 250, []int{351, 1, 227, 224, 4, 223, 6, 400, 222, 399, 398, 11, 12, 397, 14, 221, 266, 394, 18, 267, 391, 390, 218, 215, 24, 25, 388, 386, 28, 214, 30, 213, 32, 384, 381, 35, 380, 37, 378, 498, 40, 41, 376, 43, 437, 373, 46, 47, 48, 49, 371, 370, 269, 272, 367, 55, 274, 366, 58, 208, 207, 61, 277, 279, 64, 362, 360, 281, 68, 357, 282, 202, 72, 73, 354, 286, 76, 199, 198, 197, 289, 81, 349, 83, 84, 85, 193, 87, 347, 89, 345, 344, 92, 93, 291, 95, 292, 342, 189, 250, 100, 187, 186, 103, 337, 335, 106, 107, 334, 185, 184, 111, 183, 113, 114, 329, 182, 328, 157, 181, 180, 121, 179, 123, 296, 297, 126, 298, 128, 129, 130, 131, 132, 172, 171, 168, 321, 305, 138, 320, 140, 141, 309, 319, 144, 317, 316, 161, 148, 315, 150, 151, 152, 153, 311, 155, 158, 261, 230, 258, 233, 442, 236, 239, 254, 243, 244, 245, 246, 251, 404, 406, 408, 409, 413, 414, 418, 420, 422, 423, 424, 429, 435, 438, 440, 446, 447, 450, 452, 453, 458, 461, 464, 466, 468, 471, 475, 476, 478, 479, 481, 482, 485, 486, 490, 491, 493, 494, 496, 257, 287, 53, 234, 194, 29, 465, 340, 217, 411, 456, 134, 459, 332, 135, 304, 241, 226, 363, 228, 75, 80, 295, 125, 147, 137, 441, 273, 313, 265, 3, 403, 19, 256, 359, 86, 39, 368, 431, 484, 22}},
	}
	for _, c := range cases {
		got := retainBestOrder(lcgResponses(c.n, c.mod), c.keep)
		if len(got) != len(c.want) {
			t.Fatalf("n=%d keep=%d: kept %d, want %d", c.n, c.keep, len(got), len(c.want))
		}
		for i := range got {
			if got[i] != c.want[i] {
				t.Fatalf("n=%d keep=%d: order differs at %d: %d vs %d", c.n, c.keep, i, got[i], c.want[i])
			}
		}
	}
}

func TestRetainBestKeepsEverythingWhenShort(t *testing.T) {
	got := retainBestOrder([]float64{3, 1, 2}, 5)
	if len(got) != 3 || got[0] != 0 || got[1] != 1 || got[2] != 2 {
		t.Fatalf("got %v", got)
	}
}
