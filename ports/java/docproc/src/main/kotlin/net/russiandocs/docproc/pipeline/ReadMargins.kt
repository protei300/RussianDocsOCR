package net.russiandocs.docproc.pipeline

import net.russiandocs.docproc.imaging.Crop
import net.russiandocs.docproc.imaging.Image
import net.russiandocs.docproc.modules.Field
import net.russiandocs.docproc.tensors.PyNum
import kotlin.math.max
import kotlin.math.min

/**
 * Re-cuts the patch of a field labelled tight to its letters with a vertical margin, so the reading gets whole
 * glyphs. Port of `Pipeline._read_margins` (`a1153af1`).
 *
 * The box stays as the detector gave it — only the crop that is READ grows. The STS special marks are why: their
 * lines stand so close that a labelling margin merged neighbours (39-41 % overlap on real canvases), so they were
 * labelled tight, and the detector learnt to cut the tops and bottoms off the letters.
 *
 * The margin never reaches the next line: it stops halfway to the nearest box above or below that shares part of
 * the width, of any label. Single-canvas path only; the per-page path reads passports, which have no such field.
 */
public object ReadMargins {

    /**
     * Replaces, in [fields], every field the [margins] name whose crop grows. The superseded patch is closed;
     * the new one is cut from [canvas], the canvas the boxes were found on.
     */
    public fun apply(fields: MutableList<Field>, canvas: Image, margins: Map<String, Double>) {
        if (margins.isEmpty() || fields.isEmpty()) {
            return
        }
        val height = canvas.height
        for (i in fields.indices) {
            val box = fields[i].box
            val share = margins[box.label]
            if (share == null || share == 0.0) {
                continue
            }
            val x1 = box.x1.toInt()
            val y1 = box.y1.toInt()
            val x2 = box.x2.toInt()
            val y2 = box.y2.toInt()
            // `int(round((y2 - y1) * share))`: Python's round, half to even.
            val pad = PyNum.roundHalfEvenToInt((y2 - y1) * share)
            var top = max(0, y1 - pad)
            var bottom = min(height, y2 + pad)
            for (j in fields.indices) {
                val other = fields[j].box
                if (j == i || min(x2.toDouble(), other.x2) <= max(x1.toDouble(), other.x1)) {
                    continue                                         // no shared width
                }
                if (other.y2 <= y1) {                                // a line above
                    top = max(top, Math.floorDiv(other.y2.toInt() + y1 + 1, 2))
                } else if (other.y1 >= y2) {                         // a line below
                    bottom = min(bottom, Math.floorDiv(y2 + other.y1.toInt(), 2))
                }
            }
            if (top == y1 && bottom == y2) {
                continue
            }
            val patch = Crop.clampedCrop(canvas, x1, top, x2, bottom)
            fields[i].patch.close()
            // The frame moves with the crop: the patch now starts at the margin's first row.
            fields[i] = Field(box, patch, net.russiandocs.docproc.geometry.Offset(-x1.toDouble(), -top.toDouble()))
        }
    }
}
