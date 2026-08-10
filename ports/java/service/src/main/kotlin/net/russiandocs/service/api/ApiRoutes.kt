package net.russiandocs.service.api

import jakarta.servlet.http.HttpServletRequest
import jakarta.servlet.http.HttpServletResponse
import org.springframework.http.ResponseEntity
import org.springframework.web.bind.annotation.DeleteMapping
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.PathVariable
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.PutMapping
import org.springframework.web.bind.annotation.RequestParam
import org.springframework.web.bind.annotation.RestController
import org.springframework.web.multipart.MultipartFile

/**
 * The routing table.
 *
 * **Read it as a PERMISSION LIST** — `auth::requireSession` versus `auth::requireApiOrSession` says who may
 * call what, at the place the route is declared. That is the property FastAPI's `Depends` provides and the
 * reason [ApiServer.guard] is a wrapper rather than a check inside each handler: it becomes impossible to
 * forget, and visible where it matters.
 *
 * Every method is two lines: the mapping and one `guard` call. All logic lives in [ApiServer] and its
 * extension files, so this class can be read as a table and nothing else.
 */
@RestController
public class ApiRoutes(private val api: ApiServer) {

    // --- auth: no credential required, obviously ---------------------------

    @PostMapping("${ApiServer.PREFIX}/auth/pin-login")
    public fun pinLogin(request: HttpServletRequest): ResponseEntity<*> = api.pinLogin(request)

    // --- documents: API key OR session ------------------------------------
    // The same routes serve the bundled SPA and third-party integrations, which is why they accept either
    // credential rather than being duplicated per audience.

    @PostMapping("${ApiServer.PREFIX}/documents")
    public fun upload(
        request: HttpServletRequest,
        response: HttpServletResponse,
        @RequestParam(value = "file", required = false) file: MultipartFile?,
    ): ResponseEntity<*> =
        api.guard(request, response, api.auth::requireApiOrSession) { api.upload(file) }

    @GetMapping("${ApiServer.PREFIX}/documents")
    public fun list(request: HttpServletRequest, response: HttpServletResponse): ResponseEntity<*> =
        api.guard(request, response, api.auth::requireApiOrSession) { api.list(request) }

    @PostMapping("${ApiServer.PREFIX}/documents/purge")
    public fun purge(request: HttpServletRequest, response: HttpServletResponse): ResponseEntity<*> =
        api.guard(request, response, api.auth::requireSession) { api.purge() }

    @GetMapping("${ApiServer.PREFIX}/documents/{id}")
    public fun getDocument(
        request: HttpServletRequest,
        response: HttpServletResponse,
        @PathVariable id: String,
    ): ResponseEntity<*> = api.guard(request, response, api.auth::requireApiOrSession) {
        api.getDocument(api.parseId(id))
    }

    @DeleteMapping("${ApiServer.PREFIX}/documents/{id}")
    public fun deleteDocument(
        request: HttpServletRequest,
        response: HttpServletResponse,
        @PathVariable id: String,
    ): ResponseEntity<*> = api.guard(request, response, api.auth::requireApiOrSession) {
        api.deleteDocument(api.parseId(id))
    }

    @GetMapping("${ApiServer.PREFIX}/documents/{id}/progress")
    public fun progress(
        request: HttpServletRequest,
        response: HttpServletResponse,
        @PathVariable id: String,
    ): ResponseEntity<*> = api.guard(request, response, api.auth::requireApiOrSession) {
        api.documentProgress(api.parseId(id))
    }

    @PostMapping("${ApiServer.PREFIX}/documents/{id}/reprocess")
    public fun reprocess(
        request: HttpServletRequest,
        response: HttpServletResponse,
        @PathVariable id: String,
    ): ResponseEntity<*> = api.guard(request, response, api.auth::requireApiOrSession) {
        api.reprocess(api.parseId(id))
    }

    @GetMapping("${ApiServer.PREFIX}/documents/{id}/image/{kind}")
    public fun image(
        request: HttpServletRequest,
        response: HttpServletResponse,
        @PathVariable id: String,
        @PathVariable kind: String,
    ): ResponseEntity<*> = api.guard(request, response, api.auth::requireApiOrSession) {
        api.imageArtifact(api.parseId(id), kind)
    }

    // --- operator surface: session only -----------------------------------
    // An integration has no business managing keys, settings or logs, so these do not accept an API key at
    // all.

    @GetMapping("${ApiServer.PREFIX}/api-keys")
    public fun listKeys(request: HttpServletRequest, response: HttpServletResponse): ResponseEntity<*> =
        api.guard(request, response, api.auth::requireSession) { api.listKeys() }

    @PostMapping("${ApiServer.PREFIX}/api-keys")
    public fun createKey(request: HttpServletRequest, response: HttpServletResponse): ResponseEntity<*> =
        api.guard(request, response, api.auth::requireSession) { api.createKey(request) }

    @DeleteMapping("${ApiServer.PREFIX}/api-keys/{id}")
    public fun deleteKey(
        request: HttpServletRequest,
        response: HttpServletResponse,
        @PathVariable id: String,
    ): ResponseEntity<*> = api.guard(request, response, api.auth::requireSession) {
        api.deleteKey(api.parseId(id))
    }

    @GetMapping("${ApiServer.PREFIX}/settings")
    public fun getSettings(request: HttpServletRequest, response: HttpServletResponse): ResponseEntity<*> =
        api.guard(request, response, api.auth::requireSession) { api.getSettings() }

    @PutMapping("${ApiServer.PREFIX}/settings")
    public fun putSettings(request: HttpServletRequest, response: HttpServletResponse): ResponseEntity<*> =
        api.guard(request, response, api.auth::requireSession) { api.putSettings(request) }

    @GetMapping("${ApiServer.PREFIX}/logs")
    public fun logs(request: HttpServletRequest, response: HttpServletResponse): ResponseEntity<*> =
        api.guard(request, response, api.auth::requireSession) { api.logs(request) }

    @GetMapping("${ApiServer.PREFIX}/status")
    public fun status(request: HttpServletRequest, response: HttpServletResponse): ResponseEntity<*> =
        api.guard(request, response, api.auth::requireSession) { api.status() }

    // --- health: no prefix, no auth, for the container ---------------------

    @GetMapping("/health")
    public fun health(): ResponseEntity<*> = api.health()

    // --- the SPA, as a catch-all ------------------------------------------
    //
    // **`/**` — and the pattern matters.** Spring resolves the most specific mapping first, so the API
    // routes above win over this one; what this must NOT do is exclude paths that look like files. The .NET
    // port hit exactly that: its parameterless fallback carried a `nonfile` constraint, so `/` returned
    // index.html with a 200 while every hashed asset 404'd and the page rendered BLANK with no server-side
    // error anywhere.
    @GetMapping("/**")
    public fun spa(request: HttpServletRequest): ResponseEntity<*> = api.spa(request)
}
