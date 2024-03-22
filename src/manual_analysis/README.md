# Manual analysis

To run the manual analysis, use the script `manual_patch_analysis.py`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Example for analysing IR1xOR1 results
python manual_patch_analysis.py ../../results/defects4j/repairllama/lora/RepairLLaMA_defects4j_f2f_bugs_results_ir1_or1.jsonl d4j_iror1_andre.jsonl
```