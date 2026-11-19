"""The explanation step: the reply schema and its span-grounding check (explanations), the prompt,
the SDK call and the cache (explain), one structured-output request (structured), and the spend
meter every model call is checked against (spend). Model calls happen here and in the
provisional precision labels only. Imports core, data, text, match and align.
See DESIGN.md#explanations.
"""
