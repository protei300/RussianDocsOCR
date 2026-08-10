// Shared build configuration for every module in the port.
//
// The analogue of Directory.Build.props in the .NET port. Settings live here rather than being
// repeated per module, because that is how two modules quietly end up on different compiler options.

plugins {
    alias(libs.plugins.kotlinJvm) apply false
    alias(libs.plugins.kotlinSerialization) apply false
}

allprojects {
    group = "net.russiandocs"
    version = "3.0.2"
}

subprojects {
    apply(plugin = "org.jetbrains.kotlin.jvm")

    repositories {
        mavenCentral()
    }

    // JDK 21 LTS. A toolchain rather than a bare jvmTarget, so the build fails loudly on a machine
    // with the wrong JDK instead of compiling against whatever is on PATH and producing class files
    // that will not load.
    extensions.configure<org.jetbrains.kotlin.gradle.dsl.KotlinJvmProjectExtension>("kotlin") {
        jvmToolchain(21)
    }

    tasks.withType<org.jetbrains.kotlin.gradle.tasks.KotlinCompile>().configureEach {
        compilerOptions {
            // -Werror is deliberately NOT set. A port is a mechanical re-typing of Python, and some
            // of what it must reproduce reads as questionable Kotlin — the reference's own comments
            // explain why. Turning warnings into errors would pressure the next person into
            // "improving" behaviour the goldens depend on.
            freeCompilerArgs.addAll(
                // Explicit API mode on the library: every public declaration needs a visibility and
                // a return type. The point is not tidiness — it is that MAPPING.md pairs this port's
                // signatures with the other three, and an inferred return type cannot be compared.
                "-Xjsr305=strict",
            )
        }
    }

    tasks.withType<Test>().configureEach {
        useJUnitPlatform()

        // **The test JVM needs the native libraries on its search path, and inheriting the developer's
        // shell is not enough.** `libopencv_java4130.dll` links against the OpenCV core libraries beside
        // it, and Windows resolves those through PATH — so a test run from an IDE, or from Gradle in a
        // shell that happens not to have them, fails with "Can't find dependent libraries", which names
        // neither OpenCV nor the missing directory. Found exactly that way: the conformance CLI passed
        // while every unit test failed, because the CLI was launched from a shell that had them.
        //
        // Passing them here makes `gradlew test` self-sufficient. The two RDOCS_* variables are forwarded
        // as well, because NativeLibraries reads them at load time and the models live outside the module.
        val openCvHome = (findProperty("opencv.home") as String?)
            ?: System.getenv("RDOCS_OPENCV_HOME")
        val toolchainBin = System.getenv("RDOCS_TOOLCHAIN_BIN")
        if (openCvHome != null) {
            environment("RDOCS_OPENCV_HOME", openCvHome)
            val extra = listOfNotNull(
                file("$openCvHome/bin").takeIf { it.isDirectory }?.path,
                file("$openCvHome/lib").takeIf { it.isDirectory }?.path,
                toolchainBin,
            )
            if (extra.isNotEmpty()) {
                environment("PATH", (extra + (System.getenv("PATH") ?: ""))
                    .joinToString(File.pathSeparator))
            }
        }
        if (toolchainBin != null) {
            environment("RDOCS_TOOLCHAIN_BIN", toolchainBin)
        }
        System.getenv("RDOCS_MODELS_ROOT")?.let { environment("RDOCS_MODELS_ROOT", it) }
        // The invariant locale, and this is a CORRECTNESS setting rather than a nicety.
        // CONVENTIONS §6.16: `toString()` on a Double is locale-sensitive on the JVM exactly as in
        // .NET, so a machine whose default locale is ru-RU writes `0,904` where the wire contract
        // requires `0.904` — every float in the view model, silently, and only on that machine. An
        // English CI runner never reproduces it.
        systemProperty("user.language", "en")
        systemProperty("user.country", "US")
        testLogging {
            events("failed")
            exceptionFormat = org.gradle.api.tasks.testing.logging.TestExceptionFormat.FULL
        }
    }
}
