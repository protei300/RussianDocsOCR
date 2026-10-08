package net.russiandocs.docproc.geometry

import net.russiandocs.docproc.imaging.Pt
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min

/*
 * Where a point of a pipeline canvas lies on the input image. Port of `document_processing/geometry.py` (PR #19,
 * issue #18).
 *
 * Every stage that changes the geometry of the image it passes on describes that change as a map from its OUTPUT
 * back to its INPUT, and the pipeline chains the maps of the stages that actually ran, in the order they ran
 * (`Results.geometry`), so any point of the final canvas maps back to the image passed to `Recognizer.run`.
 *
 * COORDINATES ARE CONTINUOUS PIXELS. An image spans [0, width] x [0, height], and the pixel with index (i, j) is the
 * unit square centred at (i + 0.5, j + 0.5); box edges from the detectors are read the same way. OpenCV's warps put
 * pixel centres on integer coordinates instead, so [Homography] shifts by half a pixel on the way in and out and the
 * matrices OpenCV computed are used unchanged. `resize` and `rotate` need no shift in these coordinates.
 *
 * A MAP RECEIVES THE POINTS OF ONE SHAPE. A stitched canvas picks the piece a shape lies on by the shape's centroid,
 * so the corners of one box are never sent to different pages.
 *
 * Two traps, both from the reference: a chain is read OUTPUT -> INPUT (a map built "forward" sends a box off the
 * photo — the first port of PR #19 did exactly that), and "the way back is not known" is said ONCE for a whole run,
 * never per box.
 */

/** Corners of an axis-aligned box, clockwise from the top-left. `corners`. */
public fun corners(x0: Double, y0: Double, x1: Double, y1: Double): List<Pt> =
    listOf(Pt(x0, y0), Pt(x1, y0), Pt(x1, y1), Pt(x0, y1))

/**
 * A map from the output of a stage back to its input.
 *
 * [toInput] answers null when the way back is not known. NOT KNOWN IS NOT THE SAME AS UNCHANGED: a stage that
 * passes its image on untouched contributes no map at all ([Chain.then] with null), while a stage whose effect no
 * point map can express contributes [Unknown] and makes every answer downstream null. A box that quietly lands
 * somewhere else is worse than no box: the caller draws it over a photo and trusts it.
 */
public sealed interface PointMap {
    /** Points of the output on the input; null — the way back is not known. */
    public fun toInput(points: List<Pt>): List<Pt>?
}

/** The input resized: output = input * (sx, sy). */
public data class Scale(val sx: Double, val sy: Double) : PointMap {
    override fun toInput(points: List<Pt>): List<Pt> = points.map { Pt(it.x / sx, it.y / sy) }
}

/** The input shifted: output = input + (dx, dy); a crop is a negative shift. */
public data class Offset(val dx: Double, val dy: Double) : PointMap {
    override fun toInput(points: List<Pt>): List<Pt> = points.map { Pt(it.x - dx, it.y - dy) }
}

/** `ROTATE_90_COUNTERCLOCKWISE` applied [turns] times to a [width] x [height] input. */
public data class QuarterTurns(val width: Int, val height: Int, val turns: Int) : PointMap {
    override fun toInput(points: List<Pt>): List<Pt> {
        val widths = ArrayList<Int>()
        var w = width
        var h = height
        repeat(Math.floorMod(turns, 4)) {
            widths += w
            val t = w
            w = h
            h = t
        }
        // One turn sends (x, y) of a W-wide image to (y, W - x); the last turn is undone first.
        var out = points
        for (wi in widths.asReversed()) {
            out = out.map { Pt(wi - it.y, it.x) }
        }
        return out
    }
}

/**
 * The output of `warpPerspective` or `warpAffine` with [matrix] (input -> output). A 2x3 affine matrix is completed
 * to 3x3. The matrix is inverted here; the half-pixel shift is applied on the way in and out.
 */
public class Homography(matrix: Array<DoubleArray>) : PointMap {
    private val inverse: Array<DoubleArray>

    init {
        require(matrix.size == 2 || matrix.size == 3) { "geometry: a homography is 2x3 or 3x3" }
        val full = if (matrix.size == 2) {
            arrayOf(matrix[0], matrix[1], doubleArrayOf(0.0, 0.0, 1.0))
        } else {
            matrix
        }
        inverse = invert(full)
    }

    override fun toInput(points: List<Pt>): List<Pt> = points.map { p ->
        val x = p.x - 0.5
        val y = p.y - 0.5
        val u = inverse[0][0] * x + inverse[0][1] * y + inverse[0][2]
        val v = inverse[1][0] * x + inverse[1][1] * y + inverse[1][2]
        val w = inverse[2][0] * x + inverse[2][1] * y + inverse[2][2]
        Pt(u / w + 0.5, v / w + 0.5)
    }

    private companion object {
        fun invert(m: Array<DoubleArray>): Array<DoubleArray> {
            val a = m[0][0]; val b = m[0][1]; val c = m[0][2]
            val d = m[1][0]; val e = m[1][1]; val f = m[1][2]
            val g = m[2][0]; val h = m[2][1]; val i = m[2][2]
            val det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
            val inv = 1.0 / det
            return arrayOf(
                doubleArrayOf((e * i - f * h) * inv, (c * h - b * i) * inv, (b * f - c * e) * inv),
                doubleArrayOf((f * g - d * i) * inv, (a * i - c * g) * inv, (c * d - a * f) * inv),
                doubleArrayOf((d * h - e * g) * inv, (b * g - a * h) * inv, (a * e - b * d) * inv),
            )
        }
    }
}

/**
 * The output of `remap(img, xs, ys + v)`: a page unbent by a vertical displacement map.
 *
 * The bend map of the page registration's line dewarp is exactly this call, so the output pixel (i, j) SAMPLED the
 * input at (i, j + v[j, i]) — the map is already the way back, no matrix and no inversion needed. [v] is row-major,
 * [height] x [width], read bilinearly between pixel centres and held at the edge outside the map, the way the remap
 * replicated its border. A point of the output moves only along y.
 */
public class VerticalRemap(private val v: FloatArray, private val height: Int, private val width: Int) : PointMap {
    init {
        require(v.size == height * width) { "geometry: bend map of ${v.size} values is not ${height}x$width" }
    }

    override fun toInput(points: List<Pt>): List<Pt> = points.map { p ->
        // sample v at the output point in OpenCV's pixel-centre coordinates
        val x = (p.x - 0.5).coerceIn(0.0, width - 1.0)
        val y = (p.y - 0.5).coerceIn(0.0, height - 1.0)
        val x0 = floor(x).toInt()
        val y0 = floor(y).toInt()
        val x1 = min(x0 + 1, width - 1)
        val y1 = min(y0 + 1, height - 1)
        val fx = x - x0
        val fy = y - y0
        fun at(row: Int, col: Int): Double = v[row * width + col].toDouble()
        val shift = at(y0, x0) * (1 - fx) * (1 - fy) + at(y0, x1) * fx * (1 - fy) +
            at(y1, x0) * (1 - fx) * fy + at(y1, x1) * fx * fy
        Pt(p.x, p.y + shift)
    }
}

/**
 * A stage that changed the image in a way no point map expresses. The canvas is still correct and recognition is
 * unaffected — only the way back is gone, and it stays gone for every stage after this one. (The bend map of the
 * page registration is NOT such a stage: it keeps its map, see [VerticalRemap].)
 */
public object Unknown : PointMap {
    override fun toInput(points: List<Pt>): List<Pt>? = null
}

/** Maps of stages in the order the stages ran: the first one reads the input. */
public class Chain(public val maps: List<PointMap> = emptyList()) : PointMap {

    /** This chain followed by one more stage; null — the stage passed its input on unchanged. */
    public fun then(later: PointMap?): Chain = if (later == null) this else Chain(maps + later)

    override fun toInput(points: List<Pt>): List<Pt>? {
        var out = points
        for (map in maps.asReversed()) {
            // One stage that cannot answer ends the walk: the stages before it are fine, but their input is no
            // longer known.
            out = map.toInput(out) ?: return null
        }
        return out
    }
}

/** A canvas made of pieces: a rectangle (x0, y0, x1, y1) of the output and the map of what fills it. */
public class Pieces(public val pieces: List<Piece>) : PointMap {

    public class Piece(public val x0: Double, public val y0: Double, public val x1: Double, public val y1: Double,
        public val map: PointMap)

    override fun toInput(points: List<Pt>): List<Pt>? {
        var cx = 0.0
        var cy = 0.0
        for (p in points) {
            cx += p.x
            cy += p.y
        }
        cx /= points.size
        cy /= points.size

        fun distance(piece: Piece): Double {
            val dx = max(max(piece.x0 - cx, 0.0), cx - piece.x1)
            val dy = max(max(piece.y0 - cy, 0.0), cy - piece.y1)
            return dx * dx + dy * dy
        }

        // the FIRST nearest piece on a tie, as Python's `min` returns it
        var best = pieces[0]
        var bestDistance = distance(best)
        for (piece in pieces.drop(1)) {
            val d = distance(piece)
            if (d < bestDistance) {
                best = piece
                bestDistance = d
            }
        }
        return best.map.toInput(points)   // null if that piece cannot answer
    }
}

/**
 * Where a field patch lies on the canvas: the map from the patch as CUT (before the series/number turn) to the
 * canvas, and the patch size then. `FieldFrames` entries.
 */
public class FieldFrame(public val toCanvas: PointMap, public val width: Int, public val height: Int)
