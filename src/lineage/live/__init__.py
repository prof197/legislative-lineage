"""The live track: each day's new and changed bills from Open States' API (intake, openstates_api),
applied to the tables in both directions (updates' contracts, incremental), held to a full
rebuild (rebuild), the pages and feed a day touches (publish, feed), the scheduled jobs
(schedule) and the gated deploy (deploy). The top layer below the CLI. See DESIGN.md#updates.
"""
