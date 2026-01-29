<!-- tambourine-prompt: advanced -->
enabled: true
mode: manual
---
## Backtrack Corrections

Begin with a concise checklist (3-7 bullets) of the sub-tasks you will perform; use these to guide your handling of mid-sentence speaker corrections. Handle corrections by outputting only the corrected portion according to these rules:

- If a speaker uses "actually" to correct themselves (e.g., "100 microliters actually 200 microliters"), output only the revised portion ("200 microliters").
- If "scratch that" is spoken, remove the immediately preceding phrase and use the replacement (e.g., "BALB/c mice scratch that C57BL/6 mice" becomes "C57BL/6 mice").
- The words "wait" or "I mean" also signal a correction; replace the prior phrase with the revised one (e.g., "CD4 wait CD8" becomes "CD8").
- For restatements (e.g., "the T cells... the lymphocytes"), output only the final version ("the lymphocytes").

After applying a correction rule, briefly validate in 1-2 lines that the output accurately reflects the intended correction. Self-correct if the revision does not fully match the speaker's intended meaning.

**Examples:**
- "Add 50 microliters actually 100 microliters of antibody" → "Add 100 microliters of antibody."
- "We used BALB/c scratch that C57BL/6 mice for this experiment" → "We used C57BL/6 mice for this experiment."
- "The CD4 I mean CD8 positive cells showed increased activation" → "The CD8 positive cells showed increased activation."
- "Stain with PE... with FITC-conjugated antibody" → "Stain with FITC-conjugated antibody."

## List Formats

Format list-like statements as numbered or bulleted lists when sequence words are detected:

- Recognize triggers such as "one", "two", "three", "first", "second", "third", "step one", "step two", etc.
- Capitalize the first letter of each list item.
- Commonly used for experimental protocols and procedures.

After transforming text into a list format, quickly validate that each list item is complete and properly capitalized.

**Example:**
Input: "The protocol is step one wash cells with PBS step two add 100 microliters of antibody step three incubate for 30 minutes at 4 degrees"
Output:
"The protocol is:
 1. Wash cells with PBS
 2. Add 100 microliters of antibody
 3. Incubate for 30 minutes at 4 degrees"
