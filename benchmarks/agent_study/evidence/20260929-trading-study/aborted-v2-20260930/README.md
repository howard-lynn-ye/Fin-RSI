# Aborted v2 trading jobs (2026-09-30)

Jobs 1782374 (7B), 1782375 (14B) and 1782378 (32B) ran on L40S for about 7 minutes and were
cancelled by the study operator after 8 decisions in total. Reason: under the one-JSON-object
tool format, 7B wrote run_python code inside Python triple quotes (invalid JSON) on every
turn and never submitted, while tools that need no code were unaffected; this penalised the
arm that must write code. No return was computed before cancelling. The records are kept
verbatim; v3 adds a code-fence option and a lenient run_python parse for both arms.
The v1 jobs (1767302-1767304) were cancelled while pending, with no decisions.
