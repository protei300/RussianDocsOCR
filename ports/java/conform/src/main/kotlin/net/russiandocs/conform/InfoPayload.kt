package net.russiandocs.conform

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/**
 * The `info` payload, exactly as `conformance/spec/cli.md` defines it.
 *
 * **Every wire name is written by hand with `@SerialName`.** Four languages have four default naming
 * policies, and the checker reads these by name — so an inferred name is a divergence waiting for the
 * first `stages_implemented` that arrives as `stagesImplemented` and is silently absent instead.
 */
@Serializable
public data class InfoPayload(
    @SerialName("port") val port: String,
    @SerialName("language") val language: String,
    @SerialName("versions") val versions: Map<String, String>,
    @SerialName("device") val device: String,
    @SerialName("ocr_device") val ocrDevice: String,
    @SerialName("providers") val providers: List<String>,
    @SerialName("model_format") val modelFormat: String,
    @SerialName("ocr_mode") val ocrMode: String,
    @SerialName("stages_implemented") val stagesImplemented: List<String>,
    @SerialName("commit") val commit: String,
)
