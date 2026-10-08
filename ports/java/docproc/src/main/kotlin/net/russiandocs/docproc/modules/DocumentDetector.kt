package net.russiandocs.docproc.modules

import net.russiandocs.docproc.config.ModelPaths
import net.russiandocs.docproc.imaging.Image
import net.russiandocs.docproc.models.DetectionModel
import net.russiandocs.docproc.pipeline.Device
import net.russiandocs.docproc.postprocess.Box
import java.io.File
import kotlin.math.max
import kotlin.math.min

/** One box of the document detector: the corners and the confidence. Immutable — a scaled copy is a new one. */
public data class DocBox(
    public val x1: Double,
    public val y1: Double,
    public val x2: Double,
    public val y2: Double,
    public val conf: Double,
) {
    /** The same box on an image scaled by [sx] horizontally and [sy] vertically. */
    public fun scaled(sx: Double, sy: Double): DocBox = DocBox(x1 * sx, y1 * sy, x2 * sx, y2 * sy, conf)
}

/**
 * One document found in a frame, with the pages of a passport spread that lie inside it.
 * The `{'box', 'conf', 'pages': [{'box', 'conf'}]}` dict of the reference's `group_documents`.
 */
public data class DetectedDocument(public val box: DocBox, public val pages: List<DocBox>) {
    /** The same document on an image scaled by [sx], [sy]; the pages are scaled with it. */
    public fun scaled(sx: Double, sy: Double): DetectedDocument =
        DetectedDocument(box.scaled(sx, sy), pages.map { it.scaled(sx, sy) })
}

/**
 * Finds every document lying in a frame, and the pages of a passport spread. Port of
 * `pipeline_modules/document_detector/document_detector.py`.
 *
 * The first stage of the pipeline (decision №142): the type classifier, the border detector and everything
 * after them read the crop of one document instead of the whole frame. A whole frame misleads the
 * classifier whenever the document is a small part of it — a licence on an A4 scan was typed from the white
 * sheet around it (measured in the reference: 50 % by frame, 99 % by crop).
 *
 * The network is `models/DocDetect` (yolo11s, classes `document` and `page`) and it is loaded through the
 * ordinary `model.json` dispatch — `PerClassYOLODetector`, the same decoder the field detector uses.
 */
public class DocumentDetector private constructor(private val model: DetectionModel) : AutoCloseable {

    /**
     * Documents in [image], largest first, with the boxes on [image] itself.
     *
     * The caller passes the frame resized to the processing size and scales the boxes back to the input
     * photo (`Pipeline._find_documents`), which is where the full-resolution crop is cut.
     */
    public fun predict(image: Image): List<DetectedDocument> = groupDocuments(model.predict(image))

    override fun close(): Unit = model.close()

    public companion object {
        /**
         * Opens the detector, or returns null when this weight set has none.
         *
         * The reference catches `FileNotFoundError` and reads whole frames "rather than refuse to start":
         * a weight set of models-v8 or older has no `DocDetect`. Here the absence of the directory (or of
         * its `model.json`/`model.onnx`) is the same condition, said once, on stderr — `System.out` belongs
         * to the conformance payload.
         */
        public fun openOrNull(
            root: String,
            paths: Map<String, String>,
            device: Device,
            threads: Int,
        ): DocumentDetector? {
            val dir = paths["DocumentDetector"]?.let { File(ModelPaths.resolve(root, paths, "DocumentDetector"), "ONNX") }
            if (dir == null || !File(dir, "model.json").isFile || !File(dir, "model.onnx").isFile) {
                System.err.println(
                    "[!] DocumentDetector weights not found (models/DocDetect): reading whole frames. " +
                        "Run scripts/fetch_models.py for a weight set that has them.",
                )
                return null
            }
            return DocumentDetector(DetectionModel(dir.path, device, threads, root))
        }

        private fun area(b: DoubleArray): Double = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

        /** True when at least [share] of [inner] lies within [outer]. `_inside`. */
        private fun inside(inner: DoubleArray, outer: DoubleArray, share: Double = 0.7): Boolean {
            val w = min(inner[2], outer[2]) - max(inner[0], outer[0])
            val h = min(inner[3], outer[3]) - max(inner[1], outer[1])
            if (w <= 0 || h <= 0) {
                return false
            }
            return w * h >= share * max(area(inner), 1e-9)
        }

        private fun corners(b: Box): DoubleArray = doubleArrayOf(b.x1, b.y1, b.x2, b.y2)

        /**
         * Detector boxes to documents, each with the pages that lie inside it. `group_documents`.
         *
         * The detector has two classes: `document` — whatever lies in the frame as one piece (a passport
         * spread, a card, a single visible page) — and `page`, every visible page of an internal passport.
         * A page belongs to the document it lies in (the FIRST one, largest first, that holds 70 % of it);
         * a document with no page inside is a single sheet. Documents come out largest first, which is
         * the one `process_img` reads.
         *
         * `sortedByDescending` is stable, as Python's `sort(reverse=True)` is: two documents of equal
         * area keep their detection order, and so does a page sharing a corner with another.
         */
        public fun groupDocuments(boxes: List<Box>): List<DetectedDocument> {
            val docs = boxes.filter { it.label == "document" }.sortedByDescending { area(corners(it)) }
            val pages = boxes.filter { it.label == "page" }

            val owned = docs.map { ArrayList<Box>() }
            for (page in pages) {
                val owner = docs.indices.firstOrNull { inside(corners(page), corners(docs[it])) }
                if (owner != null) {
                    owned[owner] += page
                }
            }
            return docs.indices.map { i ->
                val ordered = owned[i].sortedWith(compareBy({ it.y1 }, { it.x1 }))
                DetectedDocument(
                    DocBox(docs[i].x1, docs[i].y1, docs[i].x2, docs[i].y2, docs[i].conf),
                    ordered.map { DocBox(it.x1, it.y1, it.x2, it.y2, it.conf) },
                )
            }
        }
    }
}
