package net.russiandocs.docproc.config

import java.io.File

/**
 * Locates the model artifacts and reads `models_path.yaml`.
 *
 * Port of `document_processing/config/__init__.py` and its ROOT resolution.
 */
public object ModelPaths {

    /**
     * A UTF-8 BOM.
     *
     * Needed because `model.json` files must be read BOM-free (D-10): PowerShell's
     * `Set-Content -Encoding utf8` adds one, and a JSON parser then fails on the very first character
     * with a message about invalid syntax rather than about encoding.
     */
    public const val UTF8_BOM: Char = '﻿'

    /**
     * The directory that CONTAINS `document_processing/models`.
     *
     * `RDOCS_MODELS_ROOT` wins if set; otherwise the repository root is located by walking up from the
     * working directory. Mirrors the reference's ROOT resolution and exists for the same reason: the
     * CLI is invoked from several places and a cwd-relative path silently picks up the wrong models.
     *
     * Note the semantics of the variable, which the Go port's Docker image got wrong: it names a ROOT,
     * not the models directory. Pointed at the models directory, the service starts, serves its
     * frontend and fails every document.
     */
    public fun root(): String {
        val configured = System.getenv("RDOCS_MODELS_ROOT")
        if (!configured.isNullOrEmpty()) {
            require(isLibraryRoot(configured)) {
                "config: RDOCS_MODELS_ROOT=\"$configured\" has no document_processing/models"
            }
            return configured
        }

        // The working directory, then the location of this class's jar. On the JVM there is no direct
        // equivalent of AppContext.BaseDirectory, and the code source URL is the closest thing.
        //
        // **The code-source probe must not be able to throw**, and that is not defensive padding: inside a
        // Spring Boot fat jar the location is a NESTED url (`jar:nested:/…/rdocs-service.jar/!BOOT-INF/…`),
        // and `File(uri)` answers `IllegalArgumentException: URI is not hierarchical`. Eagerly evaluated in
        // a list literal, that exception escaped before the working-directory candidate was ever tried — so
        // the service built no pipeline at all and reported "URI is not hierarchical" as its reason for
        // having no recognition. The conformance CLI never saw it: its fat jar is flat, so the location is
        // a plain file.
        val fromJar = runCatching {
            javaClass.protectionDomain?.codeSource?.location
                ?.takeIf { it.protocol == "file" }
                ?.toURI()?.let { File(it).parentFile }
        }.getOrNull()
        val starts = listOfNotNull(File(".").absoluteFile, fromJar)
        for (start in starts) {
            var dir: File? = start.canonicalFile
            while (dir != null) {
                if (isLibraryRoot(dir.path)) {
                    return dir.path
                }
                dir = dir.parentFile
            }
        }

        throw IllegalStateException(
            "config: could not locate document_processing/models; set RDOCS_MODELS_ROOT to the " +
                "repository root",
        )
    }

    private fun isLibraryRoot(dir: String): Boolean =
        File(dir, "document_processing/models").isDirectory

    /**
     * Reads `models_path.yaml` into module name to relative path.
     *
     * A hand-written parser rather than a YAML dependency, matching the Go and .NET ports. The file is
     * fourteen lines of `Key: value` with no nesting, lists, anchors or quoting, so a parser costs one
     * dependency and buys nothing — and a dependency that can interpret this file differently in one
     * language than another is precisely what a port cannot afford.
     */
    public fun load(root: String): Map<String, String> {
        val path = File(root, "document_processing/config/models_path.yaml")
        val result = LinkedHashMap<String, String>()

        path.readLines(Charsets.UTF_8).forEachIndexed { index, raw ->
            var line = raw
            if (index == 0 && line.isNotEmpty() && line[0] == UTF8_BOM) {
                line = line.substring(1)
            }
            line = line.trim()
            if (line.isEmpty() || line.startsWith("#")) {
                return@forEachIndexed
            }

            val colon = line.indexOf(':')
            require(colon > 0) { "config: $path:${index + 1}: cannot parse \"$line\"" }

            result[line.substring(0, colon).trim()] =
                normaliseSeparators(line.substring(colon + 1).trim())
        }
        return result
    }

    /** Resolves a module's model directory to an absolute path. */
    public fun resolve(root: String, paths: Map<String, String>, module: String): String {
        val relative = paths[module]
            ?: throw NoSuchElementException("config: models_path.yaml has no entry for \"$module\"")
        return File(File(root, "document_processing"), relative).path
    }

    /**
     * Turns Windows separators into the platform's own.
     *
     * **Every port must do this.** The committed YAML contains `models\Borders`, and `model.json`
     * refers to `resources\centers.npz`; on Linux those are single filenames containing a backslash, so
     * `DocTypeAngles` fails to construct — but only in a container, never on a Windows developer's
     * machine, which is the worst possible distribution of a bug.
     */
    public fun normaliseSeparators(value: String): String =
        value.replace('\\', File.separatorChar).replace('/', File.separatorChar)
}
