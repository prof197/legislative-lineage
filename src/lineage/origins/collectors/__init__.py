"""One module per source of model legislation, each declaring SOURCE and collect().

The registry (lineage.origins.collect) finds them by name; a module's name is its source key with
underscores for hyphens. Each is owned by the lane that builds it.
"""
