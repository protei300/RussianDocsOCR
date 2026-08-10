// Two modules now, three at M9 — mirroring the Go module's two binaries and the .NET solution's two
// runnable projects.
//
// **The library/service split is ENFORCED, not merely intended:** `:service` will depend on
// `:docproc` and never the reverse. That is rule 10 of service/ml/runtime.py ("only this module
// imports the library") expressed as a build fact, and it is what keeps the service testable without
// 215 MB of models.

rootProject.name = "russiandocs"

include(":docproc")
include(":conform")
include(":service")
// :service arrived at M9.


dependencyResolutionManagement {
    // Only public repositories, and this is a hard rule rather than a default: the repository is
    // public and MIT, and a build that names a private mirror is a build nobody outside can run.
    // It also cannot be fixed later — a committed internal host is in the git history forever.
    repositories {
        mavenCentral()
    }
}

pluginManagement {
    repositories {
        gradlePluginPortal()
        mavenCentral()
    }
}
