# Chicken

**A local AI agent that sizes memory to the task, runs its agents one at a time, and gets fast answers from small quantized models.**

Chicken is an AI agent for the terminal that works directly on your files and runs entirely on your own machine.
Instead of loading one huge model and hoping it fits, Chicken treats GPU memory as a resource to be scheduled:

- **Memory sized to the task.** Each step gets exactly the model and the context it needs, and nothing more.
- **One agent at a time.** Agents never run in parallel, so each one gets the whole GPU while it works.
- **Smart memory allocation.** Models are loaded, shrunk, compacted and ejected as the task moves on.
- **Fast answers.** A small, fast model decides what happens next; bigger models are called only when the task needs them.
- **Quantized models.** Every model runs fully on the GPU, never split onto the CPU.

## Status

This repository currently publishes the **design** of Chicken. The code is being rewritten and will be released here.

Read the design: [docs/DESIGN.md](docs/DESIGN.md)

## How it works, in one picture

```text
you ──> orchestrator ──> one worker ──> report ──> orchestrator ──> next worker ──> ... ──> done
          decides          executes      checked      decides
```

A small orchestrator only decides. A planner breaks big goals into steps. Specialist workers (coder, reviewer,
researcher, explorer, vision, summarizer) each do one step with a fresh memory, return a structured report and exit.
The task state lives outside the models, so nothing is lost when a model leaves the GPU.

## License

- Code: [Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for the required credit.
- Design documents in `docs/`: [Creative Commons Attribution 4.0](docs/LICENSE).

You can use, change, share and sell Chicken, as long as you keep the credit to its author in the NOTICE file
(for the code) or credit the author with a link to this repository (for the design).
The name "Chicken" and its logo are not covered by these licenses.

Created by [alemarzosw](https://github.com/alemarzosw).
