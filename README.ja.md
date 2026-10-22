# Pseudorca

**学習を使わない LLM 実験のための実験台帳**

[English](README.md)

学習向けの実験管理ツールは、ハイパーパラメータと loss 曲線を記録します。次のような実験には向きません。
- 実験の単位が、小さな改善（プロンプトの調整、リトライ規則、推論サーバーの設定）を積み重ねたノートブックである
- チームメイト同士でノートブックを派生させ合う
- 評価のたびに結果がぶれる

Pseudorca は、Kaggle 型の LLM コンペで本当に必要な次の 2 つに答えます。

1. **その改善は、これから提出するノートブックに本当に入っているか？**
   - ノートブックを**実行せずに**読みます。
   - ノートブックが書き出すファイルや埋め込んだコードを展開してから、台帳の各改善を判定します。
   - 判定は目印の文字列とパッチ照合が先です。決まらないときは LLM に聞けます。LLM が根拠として挙げた引用は、ノートブック内に実在するかを照合します。
2. **その変更はノイズを超えているか？**
   - 問題ごとのペア比較に bootstrap の区間を付けて返します。
   - 同じノートブックの再提出から、LB のブレも推定します。

データはすべて、チームの private リポジトリ内の `.pseudorca/` に置きます。サーバーもアカウントもテレメトリもありません。

> 状態：alpha。まず Kaggle のコードコンペ向けに作っています。

## インストール

```bash
uv tool install "pseudorca[all] @ git+https://github.com/pomcho555/pseudorca"
# または
pip install "pseudorca[all] @ git+https://github.com/pomcho555/pseudorca"
```

- Python 3.11 以上が必要です。本体が依存するのは `click` と `numpy` だけです。
- オプションは次の 3 つです。
  - `llm`：LLM 判定に使う OpenAI 互換と Anthropic のクライアント
  - `kaggle`：`kaggle sync` と `kaggle pull` に使う公式 Kaggle クライアント
  - `all`：両方

## 5 分で一周

`examples/make_demo_files.py` が作る合成データで一周します。中身は、オフラインで文章題を解くソルバーのノートブック 2 版、パッチ 1 つ、結果表 2 つです。

```bash
git clone https://github.com/pomcho555/pseudorca
python3 pseudorca/examples/make_demo_files.py ~/pseudorca-demo
cd ~/pseudorca-demo && git init
pseudorca init --competition word-problems
```

または `bash pseudorca/examples/demo.sh <新しいフォルダ>` で、以下を一度に通しで実行できます。

### 1. 改善を登録する

```bash
pseudorca idea add short-cot    --title "推論指示を短く"          --status adopted \
    --fingerprint "Think step by step, briefly."
pseudorca idea add retry-parse  --title "答えが読めなければ再試行" --status adopted \
    --patch retry-parse.diff
pseudorca idea add prefix-cache --title "vLLM の prefix cache"    --status testing \
    --fingerprint "enable_prefix_caching=True"
pseudorca idea add temp-06      --title "温度 0.6"                --status rejected \
    --fingerprint 're:temperature\s*=\s*0\.6'
```

改善の判定方法は 2 つあり、片方だけでも両方でも使えます。
- **目印の文字列（fingerprint）**：入っているべき文字列を書きます。先頭に `!` を付けると「入っていてはいけない」文字列、`re:` を付けると正規表現になります。複数行の文字列も書けます。
- **パッチ**：unified diff です。各 hunk が当たっているか、元に戻っているかを調べます。

状態は `idea` → `testing` → `adopted`／`rejected`／`parked` と進みます。

### 2. ノートブックを登録する

```bash
pseudorca nb add base.ipynb --tag baseline
pseudorca nb add next.ipynb --parent base.ipynb --applies short-cot --applies retry-parse --tag candidate
pseudorca nb tree next.ipynb
```

```text
PATH                                SIZE   ORIGIN
----------------------------------  -----  ---------------------------------------------------------------
cells/cell_001.py                   98 B   cell 1
cells/cell_002.py                   44 B   cell 2
cells/cell_003.py                   623 B  cell 3
cells/cell_004.py                   205 B  cell 4
files/solver.py                     495 B  cell 2 > %%writefile solver.py
markdown/cell_000.md                54 B   cell 0
payloads/cell_003/0/utils/io.py     109 B  cell 3 > line 3 str > base64 > lzma > tar member utils/io.py
payloads/cell_003/0/utils/timer.py  171 B  cell 3 > line 3 str > base64 > lzma > tar member utils/timer.py
```

`nb tree` で、反映チェックが実際に読む中身を確認できます。
- コードセル
- `%%writefile` で書き出されたファイル
- base64＋lzma＋tar で埋め込まれた 2 つのモジュール

どれも実行はしていません。

### 3. 反映チェック

```bash
pseudorca audit latest
```

```text
audit of next.ipynb (ecadc54857)

IDEA          STATUS    VERDICT    METHOD       EVIDENCE/NOTE
------------  --------  ---------  -----------  ------------------------------------------------------------
prefix-cache  testing   ✓ present  fingerprint  cells/cell_004.py:3 …eights", enable_prefix_caching=True, g…
retry-parse   adopted   ✓ present  patch        files/solver.py:11 def solve(llm, problem, sampling, retrie…
short-cot     adopted   ✓ present  fingerprint  files/solver.py:3 …d problems. Think step by step, briefly."
temp-06       rejected  ✗ absent   fingerprint  found 0/1 (missing: re:temperature\s*=\s*0\.6)

3 present, 1 absent; 4 new verdicts recorded
✓ present  ◐ partial  ✗ absent  ? unknown  ! methods disagree  c claimed, not audited  · none  * stale
```

### 4. 提出前チェック

```bash
pseudorca check next.ipynb    # 終了コード 0
pseudorca check base.ipynb    # 終了コード 1：adopted の改善が入っていない（2 はチェック自体が実行できなかったとき）
```

```text
$ pseudorca check next.ipynb
check of next.ipynb (ecadc54857), already registered

OK (2)
  retry-parse  adopted  ✓ present  patch        files/solver.py:11 def solve(llm, problem, sampling, retrie…
  short-cot    adopted  ✓ present  fingerprint  files/solver.py:3 …d problems. Think step by step, briefly."

check passed
(exit code 0)

$ pseudorca check base.ipynb
check of base.ipynb (61a5a1c1b5), already registered

MISSING (2)
  retry-parse  adopted  ✗ absent  patch        0/1 hunks applied, 1 reverted
  short-cot    adopted  ✗ absent  fingerprint  found 0/1 (missing: Think step by step, briefly.)

check failed: 2 missing
(exit code 1)
```

### 5. 改善 × ノートブックの表

```bash
pseudorca status
```

```text
IDEA          STATUS    1 base.ipynb  2 next.ipynb
------------  --------  ------------  ------------
prefix-cache  testing   ✗             ✓
retry-parse   adopted   ✗             ✓
short-cot     adopted   ✗             ✓
temp-06       rejected  ✗             ✗

✓ present  ◐ partial  ✗ absent  ? unknown  ! methods disagree  c claimed, not audited  · none  * stale
#  NOTEBOOK    SHA         ADDED
-  ----------  ----------  ----------
1  base.ipynb  61a5a1c1b5  2026-10-22
2  next.ipynb  ecadc54857  2026-10-22
```

### 6. ノイズを超えているか

```bash
pseudorca run add runs/base.csv --nb base.ipynb --label base --protocol p60x3
pseudorca run add runs/next.csv --nb next.ipynb --label next --protocol p60x3
pseudorca compare base next
```

```text
A  label base: 60 items, 3 draws per item
B  label next: 60 items, 3 draws per item
items compared: 60 (only in A: 0, only in B: 0)
mean score: A 0.51  B 0.58
B − A: +0.07  95% CI [−0.04, +0.19]
B wins/losses/ties: 20/13/27  p = 0.137 (sign-flip permutation)
SE 0.06  MDE80 0.16
A/A on A (draws split in halves): +0.03  95% CI [−0.12, +0.19]  ok
A/A on B (draws split in halves): 0.00  95% CI [−0.15, +0.16]  ok
verdict: within noise (the 95% CI includes 0)
```

この例では、60 問 × 3 試行での +0.07 はノイズと区別できません。
- 出力には、この条件で検出できる最小の効果（MDE80）も出ます。
- MDE80 を見れば、試行や問題をあとどれだけ増やせばよいかがわかります。

### 7. リーダーボード

実際のコンペでは、`pseudorca kaggle sync` がチームの提出とスコアを Kaggle から取り込みます（読み取りのみ）。ここでは手で 3 件追加します。

```bash
pseudorca sub add 1001 --nb base.ipynb --score 0.412 --date 2026-01-05
pseudorca sub add 1002 --nb base.ipynb --score 0.398 --date 2026-01-06   # 同じノートブックをもう一度
pseudorca sub add 1003 --nb next.ipynb --score 0.437 --date 2026-01-07
pseudorca lb --diff 1001 1003   # この LB の差は LB のブレより大きいか
```

```text
A  1001  base.ipynb  public 0.412  (2026-01-05)
B  1003  next.ipynb  public 0.437  (2026-01-07)
B − A: +0.025 (public score)
z = +1.79 against the LB noise (pooled SD 0.0099 from 1 notebook and 2 submissions; z = diff / (SD·√2))
verdict: within LB noise
```

### 8. 共有する

```bash
pseudorca report                # .pseudorca/cache/report.html を作る（ローカル、git の対象外）
git add .pseudorca && git commit -m "Update ledger" && git push
```

## 用語

| | |
|---|---|
| **Idea（改善）** | ノートブックに入れたい（または入れたくない）変更。状態・担当・タグ・リンク・目印の文字列・パッチを持つ。 |
| **Notebook** | ノートブックやスクリプトの 1 版。中身の sha256 で識別し、親・作者が「入れた」とする改善・タグ・Kaggle カーネルを持てる。 |
| **Verdict（判定）** | 「改善 X はノートブック N に入っているか」に対する 1 つの方法の答え。`present`／`partial`／`absent`／`unknown` のどれか。 |
| **Run** | ローカルの結果表。（問題, 試行）ごとに 1 行で数値の指標を持ち、ノートブックと評価条件（protocol）に結び付く。 |
| **Submission** | LB への提出。public／private のスコアを持ち、ノートブックに結び付く。 |

判定の方法は `fingerprint`・`patch`・`llm`・`human`（`pseudorca verdict` で人が付ける）の 4 つです。

判定は次の順でまとめます。
1. 人の判定があればそれを使います。
2. なければ、機械的な判定どうしが一致したときにそれを使います。
3. それでも決まらなければ、照合済みの LLM 判定を使います。

判定のあとで改善の中身を変えると、古い判定には stale（`*`）が付きます。

ノートブックの指定には、sha256（全体または先頭 6 文字以上）・名前・Kaggle カーネル名・`latest`・`latest:<タグ>` が使えます。

## ノートブックをどう読むか

ノートブックは実行せず、次のような読み取り専用のテキストの木に展開します。

| パス | 中身 |
|---|---|
| `cells/cell_NNN.py` | コードセル。IPython のマジックはコメントにする（行番号は保つ） |
| `files/<path>` | `%%writefile`・`Path(...).write_text(...)`・`open(..., "w").write(...)` で書かれるファイル |
| `payloads/cell_NNN/...` | 埋め込まれたデータを展開したもの |
| `markdown/cell_NNN.md` | マークダウンセル。読むだけで、判定の根拠にはしない |

- 埋め込みは、base64・base85・hex の文字列を、zlib・gzip・xz/lzma・bz2 で解凍し、tar・zip を開くところまで展開します。
- 展開には、サイズ・段数・アーカイブ内のファイル数・展開に使うメモリの上限があります。展開しなかったものは警告として表示します。
- pickle は決して読み込みません。
- `%env K=V` や `os.environ["K"] = "V"` のような環境変数の設定も集めます。

## LLM による判定（任意）

設定は `.pseudorca/config.toml` に書きます。ローカルのモデルならコードは手元から出ません。

```toml
[llm]
provider = "openai"                      # OpenAI 互換サーバーなら何でも
model = "qwen3-coder:30b"
base_url = "http://localhost:11434/v1"   # Ollama。vLLM なら通常 http://localhost:8000/v1
```

外部のモデルを使うときは、明示的な許可が要ります。

```toml
[llm]
provider = "anthropic"                   # base_url なしの "openai" なら api.openai.com
model = "claude-sonnet-5-5"
api_key_env = "ANTHROPIC_API_KEY"        # 環境変数の「名前」。キーそのものは書かない
allow_remote = true                      # ノートブックのコードがこの提供元に送られる
```

LLM の判定は次のように動きます。
- LLM が使えるのは、展開した木に対する読み取り専用の道具（ファイル一覧・文字列検索・行の読み出し）だけです。
- LLM は答えに引用を付ける必要があります。
- 引用はノートブック内に実在するかを照合します。見つからない引用は捨て、照合できる引用のない `present` は `unknown` にします。
- 送る前に、よくある形式の秘密情報を `[REDACTED]` に置き換えます。
- OpenAI 互換の接続先が localhost なら、シェルにプロキシが設定されていても必ず直接つなぎます。外部の提供元へは通常のネットワーク設定でつなぎます。

LLM に聞く範囲と費用については次のとおりです。
- 既定（`--llm auto`）では、機械的な判定で決まらなかった改善だけを LLM に聞きます。
- 判定は（ノートブック、改善の版、モデル）ごとに記録されます。チームメイトがすでに判定していれば再利用し、二重に費用はかかりません。

## チームで使う

- 台帳はチームの **private** リポジトリに置き、チームごとに別の台帳にします。
- 変更は 1 件ずつ `.pseudorca/events/` の新しいファイルになるので、`git pull` と `git push` で衝突しません。
- 誰の変更かは `git config user.name`・`--as`・`$PSEUDORCA_ACTOR` で決まり、`pseudorca log` で誰が何をしたかを見られます。
- CI や提出前のスクリプトで `pseudorca check` を回せば、adopted の改善が抜けたまま提出するのを防げます。
- レポートは非公開にしてください。GitHub Pages は private リポジトリからでも公開されます。

## 実験結果の比較

- 結果表は CSV か JSONL です。`item` が必須、`draw` は任意で、ほかの数値の列が指標になります。
- `compare A B` の A と B には、run の id・run のラベル・ノートブック（その run をまとめる）が使えます。
- 計算は次のとおりです。
  - 両方にある問題だけを使います。
  - 問題ごとに試行を平均します。
  - 問題を引き直し、さらに問題の中で試行を引き直す 2 段の bootstrap で、差の 95% 区間を出します。
- 出力には次が含まれます。
  - 勝ち／負け／引き分けと、符号反転の並べ替え検定の p 値
  - 検出力 80% で見つけられる最小の効果
  - 各 arm の試行を半分に分けて比べる A/A のチェック
- 勝ち負けを判定するのは、区間が 0 を含まず、かつ並べ替え検定の p 値が 1 − 信頼度 より小さいときだけです。問題が 6 個未満（95% の場合）なら `too few items` と出ます。
- 同じ結果表が 2 回取り込まれていても、1 回分として数えます。
- protocol の違う run どうしは、`--force` を付けない限り比べません。
- `lb` は同じノートブックの再提出から LB のブレを推定し、`lb --diff` はある差がそのブレより大きいかを判定します。

## コマンド

| コマンド | 内容 |
|---|---|
| `init` | 今のリポジトリに `.pseudorca/` を作る（途中まであるなら補う） |
| `idea add / update / list / show` | 改善の登録と表示 |
| `nb add / update / list / show / tree` | ノートブックと、反映チェックが読む中身 |
| `audit [NB]` | ノートブックに各改善が入っているかを判定する |
| `check FILE_OR_NB` | 判定したうえで、期待の改善が抜けているか、rejected の改善が入っていれば失敗する |
| `verdict NB IDEA STATUS` | 人の判定を記録する |
| `status` | 改善 × ノートブックの表 |
| `sub add / list / link`、`kaggle sync`、`kaggle pull` | 提出と Kaggle カーネル（読み取りのみ） |
| `run add / list / show`、`compare`、`lb` | 結果とノイズ |
| `report` | 1 ファイルの HTML レポート |
| `log` | 最近の変更と、誰が変更したか |

`audit`・`check`・`status`・`log`・`lb` と各 `list` コマンドは、スクリプト用に `--json` を受け付けます。`status`・`log`・`nb list`・`sub list`・`run list`・`lb`・`report` では、`-n N` で最新 N 件、`-n 0` で全件を表示します。時刻は自分のタイムゾーンで表示されます。

各コマンドで `--help` が使えます。設計と細かい規則は [DESIGN.md](DESIGN.md)、安全面の考え方は [SECURITY.md](SECURITY.md) にあります。

## 今後

- vLLM の計測値（スループット、KV キャッシュ使用率、prefix cache のヒット）を run ごとに取り込む
- run と対話ログを Inspect AI のログ形式に書き出して閲覧できるようにする
- Kaggle カーネルの実行結果ファイルを取り込む
- 認証付きの小さなサーバー（チームのライブダッシュボード用）

## ライセンス

MIT。[LICENSE](LICENSE) を参照してください。
