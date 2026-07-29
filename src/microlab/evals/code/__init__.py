"""Code-and-tool evaluation harness: sandboxed execution of model-written Python
(executor), HumanEval/MBPP task assembly (tasks), prompt construction for base and chat
checkpoints (prompts), and the tool-call eval's registry rendering + scoring (toolcall).

The executor doubles as the reward backbone for execution-verified GRPO later, so it is
deliberately dependency-free and boring."""
