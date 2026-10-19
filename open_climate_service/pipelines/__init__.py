"""Pipelines: the managed, operational deployment of one workflow.

A pipeline references exactly one registered workflow with its argument bindings, and one
export plugin as its destination. It owns validation, the dry-run approval, the delivery mode
and policy, and the run history; the workflow owns the computation and the export plugin owns
destination rendering, checks and delivery. Every run, by hand or after a dataset update,
goes through `runner.run_pipeline`.
"""
