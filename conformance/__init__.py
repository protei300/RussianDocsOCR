"""Cross-language conformance harness for the document-recognition library.

The library is being reimplemented in Go, .NET, Kotlin and C++. This package is
how each implementation is graded against the Python reference.

Layout, and the one dependency rule that matters:

    spec/     normative documents; read stages.md first
    refcli/   the Python REFERENCE cli -- the only thing here that may import
              document_processing
    runner/   the CHECKER -- spawns processes, imports no port and no library
    cases/    golden data, one directory per document

`runner` never importing an implementation is what keeps it honest: a checker
written against a port would share that port's own float, rounding and sort bugs
with the thing it is judging. The ABI is `subprocess`.
"""
