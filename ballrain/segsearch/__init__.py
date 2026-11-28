"""Offline segment search ("automatic TAS"): from the TAS state at gate A, search key sequences in the headless engine
that reach gate B earlier than the TAS with an end state not clearly worse; replay every reported line cold and
require an exact match. docs/segsearch.md; CLI: python -m ballrain.segsearch --help."""
