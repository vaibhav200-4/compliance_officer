import json
from aggregator.aggregator_func import aggregate_results


# Load sample Judge outputs
with open("../dataa/judge_input.json", "r", encoding="utf-8") as f:
    judge_results = json.load(f)


# Run aggregator
aggregated_results = aggregate_results(judge_results)


# Print result
print(json.dumps(
    aggregated_results,
    indent=2,
    ensure_ascii=False
))