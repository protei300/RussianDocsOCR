// The conformance CLI — this port's side of conformance/spec/cli.md.
//
// One executable, three subcommands, driven by the checker with exec.

plugins {
    application
    alias(libs.plugins.kotlinSerialization)
}

dependencies {
    implementation(project(":docproc"))
    implementation(libs.kotlinxSerializationJson)

    // **The GPU artefact, so ONE build serves both devices.** Unlike .NET — where a single NuGet
    // package carries both providers — the JVM publishes them separately, and the GPU jar contains
    // the CPU provider as well. Depending on both would put two copies of the native runtime on the
    // classpath and let the loader pick.
    //
    // A host without CUDA is not an error: the device-resolution logic falls back and reports the
    // provider it actually got rather than the ones merely listed.
    implementation(libs.onnxruntimeGpu)

    testImplementation(libs.kotlinTest)
    testRuntimeOnly(libs.junitPlatformLauncher)
}

application {
    mainClass.set("net.russiandocs.conform.MainKt")
}

// **A fat jar, not the `application` plugin's start scripts**, and this is a contract requirement
// rather than a preference: conformance/ports.json names ONE executable that the checker runs with
// exec, and Gradle's start script is a shell wrapper that needs a shell — different on Windows and
// Linux, and awkward to name in a single `cmd` entry. `java -jar` on a self-contained jar is the same
// invocation everywhere.
val fatJar by tasks.registering(Jar::class) {
    archiveFileName.set("rdocs-conform.jar")
    destinationDirectory.set(layout.buildDirectory.dir("dist"))
    manifest {
        attributes["Main-Class"] = "net.russiandocs.conform.MainKt"
    }
    duplicatesStrategy = DuplicatesStrategy.EXCLUDE
    from(sourceSets.main.get().output)
    dependsOn(configurations.runtimeClasspath)
    from({
        configurations.runtimeClasspath.get()
            .filter { it.name.endsWith(".jar") }
            .map { zipTree(it) }
    }) {
        // Signature files from a signed dependency invalidate the merged jar, and the failure is a
        // SecurityException at startup that says nothing about packaging.
        exclude("META-INF/*.SF", "META-INF/*.DSA", "META-INF/*.RSA", "META-INF/MANIFEST.MF")
    }
}

tasks.named("build") {
    dependsOn(fatJar)
}
