# Manual analysis

The merged results (after Sen and André analyzed the patches) are under the directories `defects4j` and `humanevaljava`.

To analyze the patches we disagree upon, you have two scripts:
- `manual_patch_analysis_martin.py` runs on all result files apart from the ones starting with `evaluation_XXX`
- `manual_patch_analysis_baseline_martin.py` runs the result files starting with `evaluation_XXX`

To run the scripts, you should:

```bash
# Setup environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Example for analysing IR1xOR1 results
# The first argument is that file to analyse
# The second argument is the path to the output file (typically the same name but replace merged with martin)
python manual_patch_analysis.py defects4j/RepairLLaMA_defects4j_f2f_bugs_results_ir1_or1_merged.jsonl defects4j/RepairLLaMA_defects4j_f2f_bugs_results_ir1_or1_martin.jsonl

# Example for analysing files starting with evaluation_XXX
python manual_patch_analysis_baseline.py defects4j/evaluation_defects4j_zero-shot-cloze_model-best-llr_merged.jsonl defects4j/evaluation_defects4j_zero-shot-cloze_model-best-llr_martin.jsonl
```

Notes:
- If you consider a patch to be semantically equivalent, we skip the remainder of the patches for the same bug (if there are still any disagreements) to save time.
- If you break the script or terminate it before finishing the analysis, either delete the output file and restart or edit the analysis script to restart the for loop `for bug in tqdm.tqdm(bugs[XX:])` at the bug you where at (replace `XX` by the number of bugs lines the output file has).
- The script always appends the bugs to the output file, including the bugs that are already agreed upon.
