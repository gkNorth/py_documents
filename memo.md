# python関連メモ

1. python3 -m venv .venv
2. . .venv/bin/activate
3. pip install pandas sklearn matplotlib numpy

touch script.py

pip list
which python
pip freeze | grep pandas

<!-- 環境パス確認 -->
```
import sys
import pprint

pprint.pprint(sys.path)
```