> **Exercise — on `main`, no branch switching.** Implement the stub in `src/microlab/exercises/phase12_tools.py`, then run `pytest -m exercise -k phase12_tools` to grade it against the reference oracle. Your solution is tracked in git; commit it when it passes.

# START HERE — tool-call parsing, schema validation, and validity rate (Phase 12)

You're on the exercises folder on `main`. You implement the tool-call parser, schema validator, and
validity-rate metric in `src/microlab/exercises/phase12_tools.py`; `parse_final_answer` and the
`run_tool_loop` ReAct driver are already on `main` (they call into these once you've
implemented them). Differential tests grade you against `microlab.model.reference.tools`.

## 1. The format

A model emits a tool call as JSON wrapped in `<tool>...</tool>`:

```
<tool>{"name": "calc", "arguments": {"expr": "2+2"}}</tool>
```

and a schema describes what's required per tool name:

```python
SCHEMA = {"calc": {"required": ["expr"]}}
```

## 2. What you implement

```bash
/home/rje/anaconda3/bin/conda run -n microlab pytest tests/exercises/test_phase12_tools.py -v
```

1. **`parse_tool_call(text)`** — regex-extract the `<tool>...</tool>` body (`re.DOTALL`, since
   the JSON may be pretty-printed), `json.loads` it, and return the dict — but only if it
   parses AND is a dict AND has a `"name"` key; default a missing `"arguments"` to `{}`.
   Return `None` on any failure (bad JSON, no tag, no name). Real model output is messy —
   this is the boundary between "the model tried to call a tool" and "nothing to execute."
2. **`validate_tool_call(call, schema)`** — `True` iff `call["name"]` is a key in `schema` AND
   every string in `schema[name]["required"]` is present in `call["arguments"]`. This catches
   the model calling a tool that doesn't exist, or leaving out a required argument, before you
   ever try to execute it.
3. **`schema_validity_rate(outputs, schema)`** — over a batch of raw model outputs, the
   fraction that both parse (`parse_tool_call` isn't `None`) AND validate
   (`validate_tool_call` is `True`). This is the metric you'd track while fine-tuning a model
   to use tools reliably.

## 3. Why this matters

Tool use turns an LM into an agent, but only if its output is machine-parseable and
schema-correct — a model that "understands" a calculator but emits malformed JSON, or the
wrong argument names, is useless in a loop. `parse_tool_call` and `validate_tool_call` are the
two gates every generated action has to pass before an agent framework will execute it;
`schema_validity_rate` is the training-time and eval-time signal for "is this model getting
better at speaking the tool interface," independent of whether the tool calls themselves are
useful.

## 4. How it's graded

Differential tests run your three functions against `microlab.model.reference.tools` on the
same kinds of inputs the reference's own test suite uses (well-formed calls, malformed JSON,
missing name, missing required arg, mixed batches), plus known-value checks (e.g. the exact
`1/3` validity rate on a 3-output batch with one valid call). Green → ping me for the Socratic
review.
