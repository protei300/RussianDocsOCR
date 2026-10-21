package net.russiandocs.service

import java.io.File
import java.net.Socket
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.nio.file.Files
import java.util.Base64
import java.util.concurrent.CopyOnWriteArrayList
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import net.russiandocs.docproc.NativeLibraries
import net.russiandocs.docproc.imaging.Image
import net.russiandocs.docproc.imaging.Io
import net.russiandocs.docproc.pipeline.Results
import net.russiandocs.docproc.pipeline.RunOptions
import net.russiandocs.docproc.viewmodel.Canvas
import net.russiandocs.docproc.viewmodel.Field
import net.russiandocs.docproc.viewmodel.Payload
import net.russiandocs.service.api.ApiServer
import net.russiandocs.service.auth.Tokens
import net.russiandocs.service.config.ProcessingMode
import net.russiandocs.service.config.Settings
import net.russiandocs.service.errors.ServiceException
import net.russiandocs.service.logging.LogRing
import net.russiandocs.service.logging.ServiceLog
import net.russiandocs.service.ml.DocumentPipeline
import net.russiandocs.service.ml.InMemory
import net.russiandocs.service.ml.InMemoryResult
import net.russiandocs.service.ml.MemoryRecogniser
import net.russiandocs.service.ml.PipelineRuntime
import net.russiandocs.service.repositories.SettingsRepository
import net.russiandocs.service.store.FileStore
import net.russiandocs.service.worker.RecognitionWorker
import org.springframework.boot.WebApplicationType
import org.springframework.boot.builder.SpringApplicationBuilder
import org.springframework.context.ConfigurableApplicationContext

/**
 * `POST /recognize` and `PROCESSING_MODE=memory` (ports/MEMORY.md §7): what a black box cannot see.
 *
 * Every test checks a way something could be KEPT, not merely that the route answers: a temporary file while
 * the upload is parsed, a buffer left holding the image, a pipeline result left holding pixels, a document
 * route that parses its body before refusing it. Recognition is replaced by a fake (no models); decoding is
 * real, so the OpenCV natives are loaded. The only image used is an anonymised repository sample.
 */
class RecognizeTests {

    private companion object {
        const val P = ApiServer.PREFIX
        const val KEY = "rdk_recognize_test_key"
        const val SECRET_TEXT = "ИВАНОВ ЛЕНИНА 17"
        const val FILENAME = "Ivanov_passport_scan.jpg"

        init {
            NativeLibraries.load()
        }

        /** An anonymised repository sample — never anything from outside the repository. */
        val SAMPLE: ByteArray by lazy {
            var dir: File? = File(".").canonicalFile
            while (dir != null && !File(dir, "samples/DL_2011/1_CR_DL_2010.jpg").isFile) {
                dir = dir.parentFile
            }
            File(requireNotNull(dir) { "samples/ not found above the working directory" },
                "samples/DL_2011/1_CR_DL_2010.jpg").readBytes()
        }

        fun allZero(buffer: ByteArray): Boolean = buffer.all { it == 0.toByte() }
    }

    private var context: ConfigurableApplicationContext? = null
    private lateinit var api: ApiServer
    private lateinit var dataDir: File
    private var port = 0
    private val client = HttpClient.newBuilder().proxy(java.net.ProxySelector.of(null)).build()
    private val json = Json { ignoreUnknownKeys = true }

    /** What the fake recogniser was handed: an alias of the pixels, and the PNG it returned. */
    private val pixels = CopyOnWriteArrayList<Image>()
    private val pngs = CopyOnWriteArrayList<ByteArray>()
    private val buffers = CopyOnWriteArrayList<ByteArray>()
    private var leaseTimeoutMs = 0L

    private val fakeRecogniser = MemoryRecogniser { rgb, _, _, withCanvas, leaseMs ->
        // An alias keeps the pixels alive past the handler's release, so the test can see whether they were
        // zeroed first — the reference's fake keeps a reference to the numpy array for the same reason.
        pixels += rgb.alias()
        leaseTimeoutMs = leaseMs
        val png = if (withCanvas) "\u0089PNG fake canvas bytes".toByteArray(Charsets.ISO_8859_1) else null
        png?.let { pngs += it }
        InMemoryResult(Payload(
            docType = "INTPASSPORT_2011", docTypeBase = "INTPASSPORT", docTypeEra = "2011", recognised = true,
            device = "cpu", canvas = Canvas(10, 8, false),
            fields = listOf(Field(name = "Address", value = SECRET_TEXT)),
            ocr = mapOf("Address" to SECRET_TEXT),
            quality = mapOf("DocConf" to JsonPrimitive(0.97), "Blur" to JsonPrimitive("good")),
        ), png)
    }

    private fun start(
        mode: String = "",
        maxUploadMb: Int = 20,
        extra: Map<String, Any> = emptyMap(),
        recogniser: MemoryRecogniser = fakeRecogniser,
    ) {
        // The bootstrap key is resolved once per process; forget whatever another test class resolved.
        Tokens.resetDefaultKeyForTests()
        dataDir = Files.createTempDirectory("rdocs-recognize").toFile()
        val cfg = Settings(jwtSecret = "recognize-test-secret", defaultApiKey = KEY, dataDir = dataDir.path,
            dataWipeOnStart = false, seedSamples = -1, processingMode = mode, maxUploadMb = maxUploadMb,
            recognizeWaitSec = 7)
        val log = ServiceLog("test")
        val db = FileStore(dataDir.path, log.sink())
        val runtime = PipelineRuntime(log.sink())
        val settings = SettingsRepository(cfg, log.sink())
        api = ApiServer(db, runtime, RecognitionWorker(db, runtime, cfg, settings, log), cfg, settings, null, log)
        api.memoryRecogniser = recogniser
        api.bufferObserver = { buffers += it }
        context = SpringApplicationBuilder(Application::class.java)
            .web(WebApplicationType.SERVLET)
            .bannerMode(org.springframework.boot.Banner.Mode.OFF)
            .properties(mapOf<String, Any>("server.port" to 0) + extra)
            .initializers({ ctx -> ctx.beanFactory.registerSingleton("apiServer", api) })
            .run()
        port = context!!.environment.getProperty("local.server.port")!!.toInt()
    }

    @AfterTest
    fun stop() {
        context?.close()
        Tokens.resetDefaultKeyForTests()
        pixels.forEach { it.close() }
        if (::dataDir.isInitialized) {
            dataDir.deleteRecursively()
        }
    }

    private fun post(path: String, body: ByteArray, contentType: String, key: String? = KEY): HttpResponse<String> {
        var builder = HttpRequest.newBuilder(URI.create("http://127.0.0.1:$port$path"))
            .header("Content-Type", contentType)
            .POST(HttpRequest.BodyPublishers.ofByteArray(body))
        key?.let { builder = builder.header("X-API-Key", it) }
        return client.send(builder.build(), HttpResponse.BodyHandlers.ofString())
    }

    private fun get(path: String): HttpResponse<String> = client.send(
        HttpRequest.newBuilder(URI.create("http://127.0.0.1:$port$path")).header("X-API-Key", KEY).GET().build(),
        HttpResponse.BodyHandlers.ofString())

    private fun multipart(image: ByteArray, boundary: String = "rdocs-test-boundary"): Pair<ByteArray, String> {
        val head = "--$boundary\r\nContent-Disposition: form-data; name=\"file\"; filename=\"$FILENAME\"\r\n" +
            "Content-Type: image/jpeg\r\n\r\n"
        val tail = "\r\n--$boundary--\r\n"
        return (head.toByteArray() + image + tail.toByteArray()) to "multipart/form-data; boundary=$boundary"
    }

    private fun obj(text: String): JsonObject = json.parseToJsonElement(text).jsonObject

    private fun logText(): String =
        LogRing.recent(2000, "", "").joinToString("\n") { "${it.message} ${it.exc ?: ""}" }

    // --------------------------------------------------------------------------- §7.1 no spooling

    @Test
    fun `a multipart body over 2 MB is split in memory and never spooled to disk`() {
        // The multipart location is a FILE, so any attempt by Tomcat to write a part fails the request: the
        // only way this answers 200 is that nothing ever asked Tomcat to parse the parts.
        val notADirectory = Files.createTempFile("rdocs-no-spool", ".file").toFile()
        try {
            start(extra = mapOf(
                "spring.servlet.multipart.location" to notADirectory.path,
                "spring.servlet.multipart.file-size-threshold" to "0B",
            ))
            // A real JPEG followed by padding: decodes like the sample, weighs more than any spool threshold.
            val big = SAMPLE + ByteArray(2_500_000)
            val (body, contentType) = multipart(big)
            assertTrue(body.size > 2 * 1024 * 1024)
            val response = post("$P/recognize", body, contentType)
            assertEquals(200, response.statusCode(), response.body())
            assertEquals(SECRET_TEXT, obj(response.body())["ocr"]!!.jsonObject["Address"]!!.jsonPrimitive.content)
            assertFalse(FILENAME in logText(), "the filename reached the log")
        } finally {
            notADirectory.delete()
        }
    }

    // --------------------------------------------------------------------------- §7.2 buffers zeroed

    @Test
    fun `every buffer the request owned is zero after the answer - raw body`() {
        start()
        val response = post("$P/recognize?images=1", SAMPLE.copyOf(), "image/jpeg")
        assertEquals(200, response.statusCode(), response.body())
        assertEquals("no-store", response.headers().firstValue("Cache-Control").orElse(""))
        val body = obj(response.body())
        val png = Base64.getDecoder().decode(body["canvas"]!!.jsonObject["image_png_base64"]!!.jsonPrimitive.content)
        assertTrue(png.size > 4 && png[1] == 'P'.code.toByte(), "the canvas PNG was not returned")

        assertTrue(buffers.isNotEmpty(), "no buffer was observed")
        buffers.forEach { assertTrue(it.isNotEmpty() && allZero(it), "a body/image/PNG buffer was not wiped") }
        assertEquals(1, pixels.size)
        assertTrue(pixels[0].width > 0 && pixels[0].nonZeroCount() == 0, "the decoded pixels were not wiped")
        assertEquals(1, pngs.size)
        assertTrue(allZero(pngs[0]), "the canvas PNG was not wiped")
    }

    @Test
    fun `every buffer the request owned is zero after the answer - multipart`() {
        start()
        val (body, contentType) = multipart(SAMPLE.copyOf())
        val response = post("$P/recognize", body, contentType)
        assertEquals(200, response.statusCode(), response.body())
        // The body and the copy of the part: two separate arrays, both wiped.
        assertTrue(buffers.size >= 2, "expected the body and the image part, got ${buffers.size}")
        buffers.forEach { assertTrue(allZero(it), "a multipart buffer was not wiped") }
        assertTrue(pixels[0].nonZeroCount() == 0, "the decoded pixels were not wiped")
    }

    // --------------------------------------------------------------------------- §7.3 pipeline scrubbed

    /** Holds images the way a real result does, plus "weights" that must survive. */
    private class FakePipeline(val fail: Boolean) : DocumentPipeline {
        val weights: Image = Io.newFilled(4, 4, 3.0, 3.0, 3.0)
        val produced = mutableListOf<Image>()
        val aliases = mutableListOf<Image>()
        var results: Results? = null
        var askedToWipe: Boolean? = null

        override fun read(frame: Image, options: RunOptions): Results {
            askedToWipe = options.wipeImages
            val r = Results()
            for ((h, v) in listOf(50 to 7.0, 8 to 9.0, 40 to 5.0)) {
                val image = r.adopt(Io.newFilled(h, 30, v, v, v))
                produced += image
                aliases += image.alias()
            }
            results = r
            if (fail) {
                // What the library does when a run fails: it closes its result, zeroing as the options say.
                if (options.wipeImages) r.wipe() else r.close()
                throw IllegalStateException("boom $SECRET_TEXT")
            }
            return r
        }

        override fun viewModel(results: Results): Payload = Payload(docType = "X")
    }

    private fun checkScrubbed(fake: FakePipeline) {
        assertEquals(true, fake.askedToWipe, "the run was not asked to zero its images")
        fake.produced.forEach { assertTrue(it.isEmpty, "an image of the result is still held") }
        assertNull(fake.results!!.canvas)
        fake.aliases.forEach { assertEquals(0, it.nonZeroCount(), "an image of the result was released unwiped") }
        assertEquals(4 * 4 * 3, fake.weights.nonZeroCount(), "model state was wiped")
        (fake.aliases + fake.weights).forEach { it.close() }
    }

    @Test
    fun `the pipeline result is zeroed and dropped after a recognition`() {
        val fake = FakePipeline(fail = false)
        Io.newFilled(10, 10, 1.0, 1.0, 1.0).use { rgb ->
            val result = InMemory.recogniseOn(fake, rgb, 0.5, 1500, withCanvas = true)
            assertEquals("X", result.payload.docType)
            assertEquals(100 * 3, rgb.nonZeroCount(), "the caller's frame belongs to the caller")
        }
        checkScrubbed(fake)
    }

    @Test
    fun `the pipeline result is zeroed and dropped when the recognition throws`() {
        val fake = FakePipeline(fail = true)
        Io.newFilled(10, 10, 1.0, 1.0, 1.0).use { rgb ->
            val error = runCatching { InMemory.recogniseOn(fake, rgb, 0.5, 1500, withCanvas = true) }
                .exceptionOrNull()
            assertTrue(error is IllegalStateException)
        }
        checkScrubbed(fake)
    }

    // --------------------------------------------------------------------------- §7.4 memory mode

    @Test
    fun `memory mode answers 404 on every document route before reading the body`() {
        start(mode = "memory")
        for ((method, path) in listOf("GET" to "/documents", "GET" to "/documents/1",
            "GET" to "/documents/1/image/canvas", "DELETE" to "/documents/1", "POST" to "/documents/purge")) {
            val response = client.send(HttpRequest.newBuilder(URI.create("http://127.0.0.1:$port$P$path"))
                .header("X-API-Key", KEY).method(method, HttpRequest.BodyPublishers.noBody()).build(),
                HttpResponse.BodyHandlers.ofString())
            assertEquals(404, response.statusCode(), "$method $path")
            assertTrue("PROCESSING_MODE=memory" in response.body(), "$method $path: ${response.body()}")
        }

        // An upload that DECLARES 50 MB and sends a few bytes of it: the answer can only arrive if nothing
        // waited for — let alone parsed or spooled — the body. No credentials either: the 404 precedes auth.
        Socket("127.0.0.1", port).use { socket ->
            socket.soTimeout = 15_000
            val head = "POST $P/documents HTTP/1.1\r\nHost: 127.0.0.1\r\n" +
                "Content-Type: multipart/form-data; boundary=b\r\nContent-Length: ${50 * 1024 * 1024}\r\n\r\n" +
                "--b\r\nContent-Disposition: form-data; name=\"file\"; filename=\"$FILENAME\"\r\n\r\n"
            socket.getOutputStream().apply { write(head.toByteArray()); write(SAMPLE, 0, 1000); flush() }
            val received = java.io.ByteArrayOutputStream()
            val input = socket.getInputStream()
            val chunk = ByteArray(4096)
            try {
                while (!received.toString(Charsets.UTF_8).contains("POST /api/v1/recognize.\"}")) {
                    val n = input.read(chunk)
                    if (n < 0) {
                        break
                    }
                    received.write(chunk, 0, n)
                }
            } catch (e: java.net.SocketTimeoutException) {
                throw AssertionError("no complete answer while the body was still unsent; got: " +
                    received.toString(Charsets.UTF_8), e)
            }
            val answer = received.toString(Charsets.UTF_8)
            assertTrue(answer.startsWith("HTTP/1.1 404"), answer)
            assertTrue("PROCESSING_MODE=memory" in answer, answer)
        }

        // /recognize still works, and the mode is reported.
        assertEquals(200, post("$P/recognize", SAMPLE.copyOf(), "image/jpeg").statusCode())
        assertEquals("memory", obj(get("$P/auth/config").body())["processing_mode"]!!.jsonPrimitive.content)
        assertTrue(dataDir.walk().none { it.isFile && it.name == "record.json" }, "memory mode stored a record")
    }

    // --------------------------------------------------------------------------- §7.6 mode resolution

    @Test
    fun `mode resolution fails toward privacy`() {
        assertEquals(ProcessingMode.Resolved("stored", null), ProcessingMode.resolve(null))
        assertEquals(ProcessingMode.Resolved("stored", null), ProcessingMode.resolve(""))
        assertEquals(ProcessingMode.Resolved("stored", null), ProcessingMode.resolve("  Stored "))
        assertEquals(ProcessingMode.Resolved("memory", null), ProcessingMode.resolve("MEMORY"))
        val typo = ProcessingMode.resolve("memroy")
        assertEquals("memory", typo.mode)
        assertTrue(typo.reason!!.contains("memroy"), typo.reason)
        assertEquals("memory", ProcessingMode.resolve("off").mode)
        assertFalse(typo.storing)
    }

    @Test
    fun `a mistyped mode means memory and says why on status`() {
        start(mode = "memroy")
        assertEquals(404, get("$P/documents").statusCode())
        assertEquals("memory", obj(get("$P/auth/config").body())["processing_mode"]!!.jsonPrimitive.content)
    }

    // --------------------------------------------------------------------------- the rest of the contract

    @Test
    fun `refusals are journalled, the journal holds outcome and time only, the log one line each`() {
        start(maxUploadMb = 1)
        assertEquals(401, post("$P/recognize", SAMPLE.copyOf(), "image/jpeg", key = null).statusCode())
        assertEquals(400, post("$P/recognize", ByteArray(0), "image/jpeg").statusCode())
        assertEquals(415, post("$P/recognize", "%PDF-1.7 not an image".toByteArray(), "application/pdf").statusCode())
        assertEquals(415, post("$P/recognize", "just some text".toByteArray(), "text/plain").statusCode())
        assertEquals(413, post("$P/recognize", byteArrayOf(-1, -40, -1) + ByteArray(2 * 1024 * 1024), "image/jpeg")
            .statusCode())
        val (noImage, ct) = "--b\r\nContent-Disposition: form-data; name=\"note\"\r\n\r\nx\r\n--b--\r\n"
            .toByteArray() to "multipart/form-data; boundary=b"
        assertEquals(400, post("$P/recognize", noImage, ct).statusCode())
        assertEquals(400, post("$P/recognize", noImage, "multipart/form-data").statusCode())
        val ok = post("$P/recognize", SAMPLE.copyOf(), "image/jpeg")
        assertEquals(200, ok.statusCode())
        val answer = obj(ok.body())
        assertEquals(JsonPrimitive(false), answer["stored"])
        assertEquals(0.97, answer["doc_conf"]!!.jsonPrimitive.content.toDouble())
        assertFalse("DocConf" in answer["quality"]!!.jsonObject)
        assertFalse("image_png_base64" in answer["canvas"]!!.jsonObject)
        assertEquals(listOf("stored", "doc_type", "doc_type_base", "doc_type_era", "recognised", "doc_conf",
            "device", "canvas", "coord_space", "coord_space_note", "boxes", "fields", "ocr", "quality", "address",
            "timings", "processing_ms"), answer.keys.toList())

        val journal = obj(get("$P/recognitions").body())
        val items = journal["items"] as JsonArray
        assertEquals(7, journal["total"]!!.jsonPrimitive.int, "the 401 must not be journalled")
        items.forEach {
            assertEquals(setOf("id", "at", "outcome", "status_code", "processing_ms", "stored"), it.jsonObject.keys)
        }
        assertEquals(listOf("recognised", "rejected", "rejected", "rejected", "rejected", "rejected", "rejected"),
            items.map { it.jsonObject["outcome"]!!.jsonPrimitive.content })
        assertEquals(6, journal["counts"]!!.jsonObject["rejected"]!!.jsonPrimitive.int)

        val lines = LogRing.recent(2000, "", "[API] recognize:").map { it.message }
        assertTrue(lines.size >= 7)
        lines.takeLast(7).forEach {
            assertTrue(Regex("""\[API] recognize: outcome=\w+ status=\d+ ms=\d+""").matches(it), it)
        }
        val text = logText()
        assertFalse(SECRET_TEXT in text || "INTPASSPORT" in text, "document data reached the log")
    }

    @Test
    fun `a busy pipeline is 503 with Retry-After and the lease waits RECOGNIZE_WAIT_SEC`() {
        start(recogniser = { _, _, _, _, leaseMs ->
            leaseTimeoutMs = leaseMs
            throw ServiceException.busy("no pipeline")
        })
        val response = post("$P/recognize", SAMPLE.copyOf(), "image/jpeg")
        assertEquals(503, response.statusCode())
        assertEquals("5", response.headers().firstValue("Retry-After").orElse(""))
        assertEquals(7_000L, leaseTimeoutMs)
        val items = obj(get("$P/recognitions").body())["items"] as JsonArray
        assertEquals("busy", items[0].jsonObject["outcome"]!!.jsonPrimitive.content)
    }

    @Test
    fun `an unexpected error is 500 and logs the type name only`() {
        start(recogniser = { _, _, _, _, _ -> throw IllegalArgumentException("failed while reading $SECRET_TEXT") })
        val response = post("$P/recognize", SAMPLE.copyOf(), "image/jpeg")
        assertEquals(500, response.statusCode())
        assertFalse(SECRET_TEXT in response.body())
        val text = logText()
        assertFalse(SECRET_TEXT in text, "the exception message reached the log")
        assertTrue("[API] recognize failed: IllegalArgumentException" in text)
    }

    @Test
    fun `stored mode keeps the document routes and reports the mode`() {
        start()
        assertEquals(200, get("$P/documents").statusCode())
        assertEquals("stored", obj(get("$P/auth/config").body())["processing_mode"]!!.jsonPrimitive.content)
        assertEquals(200, post("$P/recognize", SAMPLE.copyOf(), "image/jpeg").statusCode())
        assertEquals(0, obj(get("$P/documents").body())["total"]!!.jsonPrimitive.int, "/recognize stored a document")
        assertNotNull(pixels.firstOrNull())
    }
}
