// The web service — port of service/.
//
// **Must depend on :docproc and never the reverse.** That is rule 10 of service/ml/runtime.py ("only this
// module imports the library") turned into a build fact, and it is what lets the service be tested without
// 215 MB of models.

plugins {
    alias(libs.plugins.springBoot)
    alias(libs.plugins.kotlinSerialization)
}

dependencies {
    // The BOM, imported as a platform. Without it the starters below have no version and resolve to
    // nothing — see the note in libs.versions.toml, because the error message blames the repository.
    implementation(platform(libs.springBootBom))

    implementation(project(":docproc"))
    implementation(libs.kotlinxSerializationJson)

    // **The GPU artefact, so ONE build serves both devices.** The JVM publishes CPU and GPU separately
    // (unlike .NET's single package), and the GPU jar contains the CPU provider as well. Depending on both
    // would put two copies of the native runtime on the classpath and let the loader pick.
    implementation(libs.onnxruntimeGpu)

    // Spring Boot WEB, not WebFlux. The pipeline is synchronous in every port and must stay comparable, so a
    // reactive stack would mean bridging back to blocking code at every handler — and the reference is a
    // thread-per-request service with a pool of exactly one pipeline.
    implementation(libs.springBootStarterWeb)

    // Jackson's Kotlin module: without it Spring cannot construct a data class with a primary constructor and
    // fails at runtime with a message about no default constructor. The wire format itself is produced by
    // kotlinx.serialization — see ApiJson — so Jackson only ever handles Spring's own plumbing.
    implementation(libs.jacksonModuleKotlin)

    testImplementation(libs.springBootStarterTest)
    testImplementation(libs.kotlinTest)
    testRuntimeOnly(libs.junitPlatformLauncher)
}

// The name matches the other ports' binaries, so the two deployments read alike and a compose file differs
// only in the image tag.
tasks.named<org.springframework.boot.gradle.tasks.bundling.BootJar>("bootJar") {
    archiveFileName.set("rdocs-service.jar")
    destinationDirectory.set(layout.buildDirectory.dir("dist"))
}
