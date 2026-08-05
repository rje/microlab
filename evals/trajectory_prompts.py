"""The FIXED prompt set swept across coder-1b's milestone checkpoints.

Frozen by design: the value of a trajectory is comparability, so prompts are appended
only (never edited or removed — editing one invalidates every earlier sweep's column).
Each probes a capability expected to arrive at a different point in training, so the
sweep shows WHEN form becomes idiom becomes binding becomes algorithm.

Greedy decoding, fixed token budget, so differences between checkpoints are differences
in the MODEL, not in sampling luck.
"""

PROMPTS = [
    # -- form: does structure hold at all (expected earliest)
    {"id": "fn-skeleton", "n": 48, "text": "def add(a, b):\n"},
    {"id": "class-skeleton", "n": 64,
     "text": "class Point:\n    def __init__(self, x, y):\n"},
    # -- idiom: stock patterns memorized from the corpus
    {"id": "binary-search", "n": 96,
     "text": 'def binary_search(arr, target):\n    """Return the index of target in '
             'sorted list arr, or -1 if absent."""\n'},
    {"id": "argparse", "n": 96,
     "text": 'import argparse\n\ndef main():\n    parser = argparse.ArgumentParser('
             'description="Convert CSV files to JSON")\n'},
    {"id": "fizzbuzz", "n": 96,
     "text": "# Print numbers 1 to 100, but Fizz for multiples of 3, Buzz for 5\n"
             "for i in range(1, 101):\n"},
    # -- binding: is the name defined above the name used below (the KDA-state question)
    {"id": "var-binding", "n": 64,
     "text": "def process(items):\n    results = []\n    for item in items:\n"
             "        cleaned = item.strip().lower()\n"},
    {"id": "self-binding", "n": 64,
     "text": "class Counter:\n    def __init__(self):\n        self.count = 0\n\n"
             "    def increment(self):\n"},
    # -- cross-language: the corpus is 30% non-Python code
    {"id": "rust-fn", "n": 64, "text": "fn factorial(n: u64) -> u64 {\n"},
    {"id": "sql", "n": 48,
     "text": "-- Find the ten most recent orders with their customer names\nSELECT "},
    # -- prose-to-code: docstring comprehension rather than signature comprehension
    {"id": "docstring-only", "n": 96,
     "text": '"""Read a JSON file and return the number of top-level keys."""\n'},
    # -- knowledge: non-code slices are 34% of the mix and should show up
    {"id": "markdown-doc", "n": 64,
     "text": "# Installation\n\nTo install the package, run:\n\n```bash\n"},
    {"id": "math-prose", "n": 64,
     "text": "The derivative of x^2 is "},
]
