# Manual analysis

The merged results (after Sen and André analyzed the patches) are under the directories `defects4j` and `humanevaljava`.

To analyze the patches we disagree upon, run `manual_patch_analysis_martin.py`

To run the scripts, you should:

```bash
# Setup environment
cd ./src/manual_analysis_martin/
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python manual_patch_analysis_martin.py
```

Notes:
- If you consider a patch to be semantically equivalent, we skip the remainder of the patches for the same bug (if there are still any disagreements) to save time.
