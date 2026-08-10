"""The conformance CHECKER.

Imports numpy and the standard library, and nothing else. In particular it must
never import `document_processing`, `service`, or any port: the boundary to an
implementation is `subprocess`, and a checker that shared code with the thing it
judges would share its bugs too.

    python -m conformance.runner list
    python -m conformance.runner run --port python
    python -m conformance.runner run --port go --profile gpu
"""
