// The library — port of document_processing/.
//
// **Must never depend on :service.** That is rule 10 of service/ml/runtime.py ("only this module
// imports the library") turned into a build fact, and it is what lets the service be tested without
// 215 MB of models.

plugins {
    alias(libs.plugins.kotlinSerialization)
}

dependencies {
    // **COMPILE-ONLY, so exactly ONE artefact reaches any runtime classpath.** Unlike .NET — where one
    // NuGet package carries both providers — the JVM publishes CPU and GPU as separate modules with the
    // SAME `ai.onnxruntime` classes. Declared as `implementation` here, this module's CPU artefact and a
    // consumer's GPU artefact both land on the classpath (Gradle cannot dedupe two different modules), the
    // loader picks whichever jar it reads first, and which provider a device request actually got becomes
    // classpath order. `compileOnly` makes the choice belong to the consumer, which is where it is
    // documented: :conform and :service each declare the GPU artefact, whose jar contains the CPU provider
    // as well.
    compileOnly(libs.onnxruntime)
    testImplementation(libs.onnxruntimeGpu)
    implementation(libs.kotlinxSerializationJson)

    // **OpenCV comes from a LOCAL BUILD, not from a repository, and that is not a shortcut.**
    //
    // There is no official `org.opencv` artefact on Maven Central at all — checked, not assumed:
    // all 24 `a:opencv` hits are third-party republishers and the newest stop at 4.9/4.10.
    // 4.13.0 is mandatory rather than preferred, because contour approximation changed in OpenCV
    // 4.8 and the conformance goldens encode 4.13 behaviour. A port on 4.9 fails
    // `borders.segments` for a reason nothing in the code explains.
    //
    // So the jar and its JNI library are built by `tools/build-opencv.*` and located through
    // RDOCS_OPENCV_HOME. NOT committed: the jar is small but it is build output, and the native
    // libraries beside it are 35 MB — and no port may carry binaries.
    implementation(files(openCvJar()))

    testImplementation(libs.kotlinTest)
    testRuntimeOnly(libs.junitPlatformLauncher)
}

/**
 * Locates the OpenCV jar, or fails with an instruction rather than a stack trace.
 *
 * Resolution order is property, then environment, then the conventional spike location — the same
 * shape as `ModelPaths` in the other ports, and for the same reason: the build runs from three
 * places (a developer's machine, CI, a Docker stage) with three different layouts.
 */
fun openCvJar(): String {
    val version = libs.versions.opencv.get()
    val flat = version.replace(".", "")          // 4.13.0 -> 4130, OpenCV's own jar naming
    val home = (findProperty("opencv.home") as String?)
        ?: System.getenv("RDOCS_OPENCV_HOME")

    val candidates = buildList {
        if (home != null) {
            add(file("$home/bin/opencv-$flat.jar"))
            add(file("$home/share/java/opencv4/opencv-$flat.jar"))
            add(file("$home/opencv-$flat.jar"))
        }
        add(file("${rootDir.parentFile.parentFile}/../kotlin-spike/build-java/bin/opencv-$flat.jar"))
    }
    val found = candidates.firstOrNull { it.isFile }
    if (found != null) {
        return found.absolutePath
    }
    error(
        """
        OpenCV $version was not found.

        This port builds OpenCV itself: there is no official org.opencv artefact on Maven Central,
        and 4.13.0 is required because contour approximation changed in 4.8 and the conformance
        goldens encode 4.13 behaviour.

        Build it (see ports/java/README.md for the full cmake line, kept identical to the Docker
        stage's), then point the build at the result:

            -Popencv.home=<dir>     or     RDOCS_OPENCV_HOME=<dir>

        Looked in:
        ${candidates.joinToString("\n        ") { it.path }}
        """.trimIndent(),
    )
}
