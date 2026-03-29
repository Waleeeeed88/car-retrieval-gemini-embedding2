# Human Review Checklist

Use this after running any golden task.

## Relevance

- Is the top result clearly in the right vehicle class?
- Is at least one of the top 3 clearly correct?
- Are there obvious mismatches dominating the top results?

## Images

- Does each visible result show a car image?
- Does the image match the returned vehicle?
- Are any images clearly broken, missing, or misleading?

## Specs and Finance

- Does the expanded details panel show summary, specs, and finance?
- Are the spec snippets useful for the query?
- Are the finance snippets useful for the query?

## UI Behavior

- Did the query mode switch correctly?
- Did file upload work for the chosen mode?
- Did the page render results without obvious layout breakage?
- Did any expanders, buttons, or inputs fail unexpectedly?

## Failure Signals

Flag the run if any of these happen:

- top results are the wrong vehicle class
- no result image appears
- details panel is empty or missing key sections
- upload mode fails silently
- duplicate or junk results dominate the output
