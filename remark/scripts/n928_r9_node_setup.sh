#!/bin/bash
# (ProtoQA) node setup: run once per node after /tmp/qenv is staged (training env staged from the packed uv venv).
# 1. installs nltk 3.10.3 (+ joblib, defusedxml) and scipy 1.17.1 into $PY (default /tmp/qenv/bin/python) with --no-deps,
# online first, else from the wheels on shared storage; refuses if numpy changed (qenv: numpy 2.2.6);
# 2. unpacks the pinned NLTK data (wordnet, omw-1.4, stopwords, punkt, punkt_tab; fetched 2026-10-02) into /tmp/nltk_data
# and pins corpora/stopwords/english to remark/data/protoqa/nltk_stopwords_english.txt (198 words; the WordNet
# score depends on the list);
# 3. verifies the 198-word list, a few matcher decisions, and the official-metric code path;
# 4. stages the ProtoQA datasets to /tmp/data/protoqa (datasets cannot memory-map arrow files on the shared network mount for
# multiprocess maps; the GRPO launcher reads /tmp).
# Usage: bash remark/scripts/n928_r9_node_setup.sh            (PY=python3 bash ... for the system python as well)
set -euo pipefail
R=${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
P=$R/remark/data/protoqa
PY=${PY:-/tmp/qenv/bin/python}

NP_BEFORE=$($PY -c "import numpy; print(numpy.__version__)")
if ! $PY -m pip install --no-deps -r $P/node_requirements.txt; then
  echo "[r9 setup] online pip install failed; installing the wheels from $P/wheels"
  $PY -m pip install --no-deps --no-index $P/wheels/*.whl
fi
NP_AFTER=$($PY -c "import numpy; print(numpy.__version__)")
if [ "$NP_BEFORE" != "$NP_AFTER" ]; then echo "[r9 setup] numpy changed $NP_BEFORE -> $NP_AFTER: STOP"; exit 1; fi

if [ ! -f /tmp/nltk_data/.r9_ok ]; then
  rm -rf /tmp/nltk_data.part
  $PY - "$P/nltk_data" /tmp/nltk_data.part <<'EOF'
import sys, zipfile
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
for sub, names in (("corpora", ("stopwords", "wordnet", "omw-1.4")), ("tokenizers", ("punkt", "punkt_tab"))):
    (dst / sub).mkdir(parents=True, exist_ok=True)
    for n in names:
        zipfile.ZipFile(src / sub / f"{n}.zip").extractall(dst / sub)
EOF
  cp $P/nltk_stopwords_english.txt /tmp/nltk_data.part/corpora/stopwords/english
  rm -rf /tmp/nltk_data && mv /tmp/nltk_data.part /tmp/nltk_data && touch /tmp/nltk_data/.r9_ok
fi

NLTK_DATA=/tmp/nltk_data $PY - <<EOF
import sys
sys.path.insert(0, "$R/remark")
from nltk.corpus import stopwords
from lib.r9_common import pinned_stopwords
sw = stopwords.words("english")
assert len(sw) == 198 and frozenset(sw) == pinned_stopwords(), len(sw)
from lib.r9_rewards import matches, item_rewards   # asserts the pinned list again
assert matches("peaches", ["peach"]) and matches("flying", ["fly"]) and matches("the", ["it"])
assert not matches("red car", ["car"]) and not matches("hat", ["sun hat"])
cl, cnt = [["umbrella"], ["sunscreen"], ["shade"]], [5, 3, 2]
assert item_rewards("mean", ["Umbrella", "umbrellas", "shade"], cl, cnt, -1)[:3] == [0.5, 0.0, 0.2]
from lib.r9_official import general_eval
assert abs(general_eval(["sunscreen", "umbrella"], cl, cnt, 1) - 3 / 5) < 1e-12
import nltk, numpy, scipy
print(f"[r9 setup] ok: nltk {nltk.__version__}, scipy {scipy.__version__}, numpy {numpy.__version__}, 198 stopwords, matcher checks passed")
EOF

mkdir -p /tmp/data/protoqa
for d in grpo grpo_union sft_dataset; do
  if [ ! -f /tmp/data/protoqa/$d/.staged ]; then
    rm -rf /tmp/data/protoqa/$d.part && cp -r $R/remark/data/crs/protoqa/$d /tmp/data/protoqa/$d.part \
      && touch /tmp/data/protoqa/$d.part/.staged && rm -rf /tmp/data/protoqa/$d && mv /tmp/data/protoqa/$d.part /tmp/data/protoqa/$d
  fi
done
echo "[r9 setup] datasets staged in /tmp/data/protoqa: $(ls /tmp/data/protoqa | tr '\n' ' ')"
