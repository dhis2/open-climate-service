"""Pipelines: one operator object over a sync schedule, a workflow trigger, an export and a delivery.

This first slice creates, validates and dry-runs a pipeline from a published dataset and a
registered feature collection, with a spatial reducer and no temporal resampling. It does
not activate anything: the compiled configuration is shown for an operator to apply.
"""
