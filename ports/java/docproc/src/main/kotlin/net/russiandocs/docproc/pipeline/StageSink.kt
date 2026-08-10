package net.russiandocs.docproc.pipeline

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import net.russiandocs.docproc.imaging.Image
import net.russiandocs.docproc.tensors.Dtype
import net.russiandocs.docproc.tensors.NdArray
import net.russiandocs.docproc.tensors.Npy
import java.io.File

/**
 * Receives per-stage intermediates for the conformance probe.
 *
 * Production passes [NullStageSink], which costs nothing and changes no behaviour. That matters: the
 * intermediates must NOT be threaded through return values, because that would alter the very code the
 * ports are meant to copy.
 */
public interface StageSink {
    public fun emit(stage: String, payload: JsonElement)

    /** Convenience for array stages, so a caller does not build the wrapper itself. */
    public fun emitArray(stage: String, array: NdArray)

    /** Convenience for image stages: copy the pixels out and emit as an array. */
    public fun emitImage(stage: String, image: Image): Unit = emitArray(stage, image.toArray())
}

/** The production sink. Does nothing, on purpose. */
public object NullStageSink : StageSink {
    override fun emit(stage: String, payload: JsonElement) {}
    override fun emitArray(stage: String, array: NdArray) {}
    // emitImage is overridden too, so production never pays for the pixel COPY that toArray does.
    // Inheriting the default would make the null sink quietly expensive on the image stages — which is
    // the one place where "does nothing" has to mean it.
    override fun emitImage(stage: String, image: Image) {}
}

/**
 * Writes one file per stage into a directory, plus `stages.json` as an ordered index.
 *
 * File naming and the index shape are fixed by `conformance/spec/stages.md` and must match the other
 * ports exactly — the checker reads them.
 */
public class DirectoryStageSink(
    private val root: String,
    private val upTo: String? = null,
) : StageSink {

    private val index = mutableListOf<JsonObject>()
    private var stopped = false

    private val json = Json { prettyPrint = true; explicitNulls = true; encodeDefaults = true }

    public val count: Int get() = index.size

    override fun emit(stage: String, payload: JsonElement) {
        if (stopped) {
            return
        }
        val file = safeName(stage) + ".json"
        File(root).mkdirs()
        File(root, file).writeText(json.encodeToString(JsonElement.serializer(), payload) + "\n")
        index += JsonObject(
            mapOf(
                "stage" to JsonPrimitive(stage),
                "file" to JsonPrimitive(file),
                "kind" to JsonPrimitive("json"),
                "dtype" to JsonNull,
                "shape" to JsonNull,
            ),
        )
        maybeStop(stage)
    }

    override fun emitArray(stage: String, array: NdArray) {
        if (stopped) {
            return
        }
        val file = safeName(stage) + ".npy"
        Npy.save(File(root, file).path, array)
        index += JsonObject(
            mapOf(
                "stage" to JsonPrimitive(stage),
                "file" to JsonPrimitive(file),
                "kind" to JsonPrimitive("npy"),
                "dtype" to JsonPrimitive(dtypeName(array.dtype)),
                "shape" to JsonArray(array.shape.map { JsonPrimitive(it) }),
            ),
        )
        maybeStop(stage)
    }

    /**
     * `--upto` stops AFTER the named stage, inclusive.
     *
     * This is what makes a milestone verifiable before the pipeline is finished, and it is the reason
     * the harness exists at all rather than being written at the end.
     */
    private fun maybeStop(stage: String) {
        if (!upTo.isNullOrEmpty() && stage == upTo) {
            stopped = true
        }
    }

    /**
     * Stage names contain dots but never separators; taking the file name defends the dump directory
     * against a stage name that somehow acquires one.
     */
    private fun safeName(stage: String): String = File(stage).name

    /** Writes the index. Call once, at the end. */
    public fun close() {
        File(root).mkdirs()
        val payload = JsonObject(mapOf("stages" to JsonArray(index)))
        File(root, "stages.json")
            .writeText(json.encodeToString(JsonElement.serializer(), payload) + "\n")
    }

    private fun dtypeName(dtype: Dtype): String = when (dtype) {
        Dtype.FLOAT32 -> "<f4"
        Dtype.FLOAT64 -> "<f8"
        Dtype.UINT8 -> "|u1"
        Dtype.INT64 -> "<i8"
        Dtype.UNICODE -> "<U"
    }
}
