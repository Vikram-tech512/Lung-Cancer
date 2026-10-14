"""
backend/pipeline/stages/__init__.py
=====================================
Individual pipeline stage runners.

Each stage exposes a single `run(ctx: PipelineContext) -> dict` method
(sync or async) that reads from ctx.get(...) and writes output via ctx.set(...).
"""
