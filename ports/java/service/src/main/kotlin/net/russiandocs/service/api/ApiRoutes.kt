package net.russiandocs.service.api

import jakarta.servlet.http.HttpServletRequest
import jakarta.servlet.http.HttpServletResponse
import org.springframework.http.ResponseEntity
import org.springframework.web.bind.annotation.DeleteMapping
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.PatchMapping
import org.springframework.web.bind.annotation.PathVariable
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.PutMapping
import org.springframework.web.bind.annotation.RestController
import org.springframework.web.multipart.MultipartHttpServletRequest
import org.springframework.web.util.WebUtils

/**
 * The routing table — ports/AUTH.md §5, and nothing else.
 *
 * **Read it as a PERMISSION LIST**: `api.auth.requireAdmin` versus `api.auth.requireApiOrViewer` says who may
 * call what, at the place the route is declared. That is the property FastAPI's `Depends` provides and the
 * reason [ApiServer.guard] is a wrapper rather than a check inside each handler: it becomes impossible to
 * forget, and visible where it matters. `RouteTableTests` compares this class against the AUTH.md table —
 * every route listed, nothing unlisted, each with the NAMED guard — by asking the router, not this file.
 *
 * **No handler takes a body parameter.** Spring would bind (and could reject) a `@RequestBody` or a multipart
 * part BEFORE the guard ran; every body is read inside the guard's block instead, so a viewer's upload is a
 * 403, not a 400 about the file.
 *
 * Every method is two lines: the mapping and one `guard` call. All logic lives in [ApiServer] and its
 * extension files, so this class can be read as a table and nothing else.
 */
@RestController
public class ApiRoutes(private val api: ApiServer) {

    // --- sign-in: public, obviously ---------------------------------------

    @GetMapping("${ApiServer.PREFIX}/auth/config")
    public fun authConfig(): ResponseEntity<*> = api.public { api.authConfigEndpoint() }

    @PostMapping("${ApiServer.PREFIX}/auth/pin-login")
    public fun pinLogin(request: HttpServletRequest): ResponseEntity<*> = api.public { api.pinLogin(request) }

    @PostMapping("${ApiServer.PREFIX}/auth/login")
    public fun login(request: HttpServletRequest): ResponseEntity<*> = api.public { api.login(request) }

    // --- the two routes that admit a session still owing a password change --------------------------

    @GetMapping("${ApiServer.PREFIX}/auth/me")
    public fun me(request: HttpServletRequest, response: HttpServletResponse): ResponseEntity<*> =
        api.guard(request, response, api.auth.requireSessionAllowPasswordChange) { api.whoami(it) }

    @PostMapping("${ApiServer.PREFIX}/auth/change-password")
    public fun changePassword(request: HttpServletRequest, response: HttpServletResponse): ResponseEntity<*> =
        api.guard(request, response, api.auth.requireSessionAllowPasswordChange) {
            api.changePassword(request, it)
        }

    // --- documents: API key at any level, or a session with a role ------------------------------
    // The same routes serve the bundled SPA and third-party integrations. Reads need viewer, writes operator.

    @PostMapping("${ApiServer.PREFIX}/documents")
    public fun upload(request: HttpServletRequest, response: HttpServletResponse): ResponseEntity<*> =
        api.guard(request, response, api.auth.requireApiOrOperator) {
            api.upload(WebUtils.getNativeRequest(request, MultipartHttpServletRequest::class.java)
                ?.getFile("file"))
        }

    @GetMapping("${ApiServer.PREFIX}/documents")
    public fun list(request: HttpServletRequest, response: HttpServletResponse): ResponseEntity<*> =
        api.guard(request, response, api.auth.requireApiOrViewer) { api.list(request) }

    @PostMapping("${ApiServer.PREFIX}/documents/purge")
    public fun purge(request: HttpServletRequest, response: HttpServletResponse): ResponseEntity<*> =
        api.guard(request, response, api.auth.requireAdmin) { api.purge() }

    @GetMapping("${ApiServer.PREFIX}/documents/{id}")
    public fun getDocument(
        request: HttpServletRequest,
        response: HttpServletResponse,
        @PathVariable id: String,
    ): ResponseEntity<*> = api.guard(request, response, api.auth.requireApiOrViewer) {
        api.getDocument(api.parseId(id))
    }

    @DeleteMapping("${ApiServer.PREFIX}/documents/{id}")
    public fun deleteDocument(
        request: HttpServletRequest,
        response: HttpServletResponse,
        @PathVariable id: String,
    ): ResponseEntity<*> = api.guard(request, response, api.auth.requireApiOrOperator) {
        api.deleteDocument(api.parseId(id))
    }

    @GetMapping("${ApiServer.PREFIX}/documents/{id}/progress")
    public fun progress(
        request: HttpServletRequest,
        response: HttpServletResponse,
        @PathVariable id: String,
    ): ResponseEntity<*> = api.guard(request, response, api.auth.requireApiOrViewer) {
        api.documentProgress(api.parseId(id))
    }

    @PostMapping("${ApiServer.PREFIX}/documents/{id}/reprocess")
    public fun reprocess(
        request: HttpServletRequest,
        response: HttpServletResponse,
        @PathVariable id: String,
    ): ResponseEntity<*> = api.guard(request, response, api.auth.requireApiOrOperator) {
        api.reprocess(api.parseId(id))
    }

    @GetMapping("${ApiServer.PREFIX}/documents/{id}/image/{kind}")
    public fun image(
        request: HttpServletRequest,
        response: HttpServletResponse,
        @PathVariable id: String,
        @PathVariable kind: String,
    ): ResponseEntity<*> = api.guard(request, response, api.auth.requireApiOrViewer) {
        api.imageArtifact(api.parseId(id), kind)
    }

    // --- status: any signed-in role, never an API key -------------------------------------------

    @GetMapping("${ApiServer.PREFIX}/status")
    public fun status(request: HttpServletRequest, response: HttpServletResponse): ResponseEntity<*> =
        api.guard(request, response, api.auth.requireViewer) { api.status() }

    // --- operator surface: administrators only -------------------------------------------------
    // An integration has no business managing keys, settings, logs or accounts, so none of these accepts an
    // API key at all.

    @GetMapping("${ApiServer.PREFIX}/api-keys")
    public fun listKeys(request: HttpServletRequest, response: HttpServletResponse): ResponseEntity<*> =
        api.guard(request, response, api.auth.requireAdmin) { api.listKeys() }

    @PostMapping("${ApiServer.PREFIX}/api-keys")
    public fun createKey(request: HttpServletRequest, response: HttpServletResponse): ResponseEntity<*> =
        api.guard(request, response, api.auth.requireAdmin) { api.createKey(request) }

    @DeleteMapping("${ApiServer.PREFIX}/api-keys/{id}")
    public fun deleteKey(
        request: HttpServletRequest,
        response: HttpServletResponse,
        @PathVariable id: String,
    ): ResponseEntity<*> = api.guard(request, response, api.auth.requireAdmin) {
        api.deleteKey(api.parseId(id))
    }

    @GetMapping("${ApiServer.PREFIX}/settings")
    public fun getSettings(request: HttpServletRequest, response: HttpServletResponse): ResponseEntity<*> =
        api.guard(request, response, api.auth.requireAdmin) { api.getSettings() }

    @PutMapping("${ApiServer.PREFIX}/settings")
    public fun putSettings(request: HttpServletRequest, response: HttpServletResponse): ResponseEntity<*> =
        api.guard(request, response, api.auth.requireAdmin) { api.putSettings(request) }

    @GetMapping("${ApiServer.PREFIX}/logs")
    public fun logs(request: HttpServletRequest, response: HttpServletResponse): ResponseEntity<*> =
        api.guard(request, response, api.auth.requireAdmin) { api.logs(request) }

    @GetMapping("${ApiServer.PREFIX}/users")
    public fun listUsers(request: HttpServletRequest, response: HttpServletResponse): ResponseEntity<*> =
        api.guard(request, response, api.auth.requireAdmin) { api.listUsers() }

    @PostMapping("${ApiServer.PREFIX}/users")
    public fun createUser(request: HttpServletRequest, response: HttpServletResponse): ResponseEntity<*> =
        api.guard(request, response, api.auth.requireAdmin) { api.createUser(request, it) }

    @PatchMapping("${ApiServer.PREFIX}/users/{id}")
    public fun updateUser(
        request: HttpServletRequest,
        response: HttpServletResponse,
        @PathVariable id: String,
    ): ResponseEntity<*> = api.guard(request, response, api.auth.requireAdmin) {
        api.updateUser(request, it, id)
    }

    @PostMapping("${ApiServer.PREFIX}/users/{id}/password")
    public fun resetPassword(
        request: HttpServletRequest,
        response: HttpServletResponse,
        @PathVariable id: String,
    ): ResponseEntity<*> = api.guard(request, response, api.auth.requireAdmin) {
        api.resetPassword(request, it, id)
    }

    @DeleteMapping("${ApiServer.PREFIX}/users/{id}")
    public fun deleteUser(
        request: HttpServletRequest,
        response: HttpServletResponse,
        @PathVariable id: String,
    ): ResponseEntity<*> = api.guard(request, response, api.auth.requireAdmin) { api.deleteUser(it, id) }

    // Both modes: in PIN mode the actor is the literal 'pin'. Only account management is mode-gated.
    @GetMapping("${ApiServer.PREFIX}/users/audit/entries")
    public fun auditEntries(request: HttpServletRequest, response: HttpServletResponse): ResponseEntity<*> =
        api.guard(request, response, api.auth.requireAdmin) { api.listAudit(request) }

    // --- health: no prefix, no auth, for the container ---------------------

    @GetMapping("/health")
    public fun health(): ResponseEntity<*> = api.health()

    // --- the SPA, as a catch-all ------------------------------------------
    //
    // **`/**` — and the pattern matters.** Spring resolves the most specific mapping first, so the API
    // routes above win over this one; what this must NOT do is exclude paths that look like files. The .NET
    // port hit exactly that: its parameterless fallback carried a `nonfile` constraint, so `/` returned
    // index.html with a 200 while every hashed asset 404'd and the page rendered BLANK with no server-side
    // error anywhere. It is not part of the API surface — anything under the prefix that reaches it is a
    // JSON 404.
    @GetMapping("/**")
    public fun spa(request: HttpServletRequest): ResponseEntity<*> = api.spa(request)
}
