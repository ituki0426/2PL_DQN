# EXP003：CPU・EAP による実験

更新日：2026-09-10。現在の実装仕様をまとめた文書です。設定値の正本はこのディレクトリ内のコードです。

## 目的と他の実験との違い

EXP001 の CPU 版を基に、能力推定を EAP（事後平均）に統一した実験です。DQN の学習・検証・評価、解析的な規則での能力推定、選択分析、速度測定で EAP を使います。

EXP001 と比べて、能力推定法、初期推定値、初期分散の扱いが変わります。EXP002 の MPS 対応は引き継いでおらず、DQN を含め CPU で実行します。

関連：[EXP001](../EXP001/exp_summary.md) / [EXP002](../EXP002/exp_summary.md) / [EXP003](../EXP003/exp_summary.md)

## 能力推定

事前分布は `N(0,1)`、計算範囲は −4〜4 の81点（間隔0.1）です。各格子点における2PLの尤度と事前分布の密度を掛け合わせ、正規化した重みで能力値の平均を求めます。

`EAP = Σ(格子点の能力値 × 正規化した事後重み)`

回答前の初期推定値は全員0です。正解・不正解の混合、全問正解、全問不正解のすべてに同じ EAP を適用し、1問回答するたびに更新します。

事後分散も同じ格子上の事後分布から計算します。DQN の学習時には事後平均と分散を一度の計算で取得し、初期分散も格子上の事前分布から求めます。

`existing` と `proposed` は項目選択・学習設定の名称で、本実験では両方とも能力推定に EAP を使います。参照能力値を使う報酬も、全回答から求めた EAP を参照します。

実装：[irt.py](irt.py)、[simulate.py](simulate.py)、[dqn.py](dqn.py)。

選択分析では、初回の推定値が全員0なので `slope_b_on_prev`（推定値に対する選択困難度の回帰係数）は計算できず、`NaN` として保存します。学習用の初期推定値は0ですが、EXP001 と乱数の消費順を合わせるため、旧初期値用の乱数を生成して破棄しています。

## EXP001 から変更したプログラムの箇所

EXP001 の Python ファイルを EXP003 にコピーしたうえで、以下を変更しています。リンク先は EXP003 内の実装です。比較元は [EXP001 ディレクトリ](../EXP001/) の同名ファイルです。

| ファイル・関数 | EXP001 の処理 | EXP003 での変更と目的 |
|---|---|---|
| [irt.py](irt.py)：`mle()` → `eap()` | 混合反応には二分法による MLE、全問正解・全問不正解には Dodd 規則を適用 | `mle()` を `eap(bank, items, resp)` に置き換え。未回答なら0、回答があれば既存の `posterior()` の第1戻り値である事後平均を返す。前回推定値と二分法の反復回数の引数も不要になった。 |
| [simulate.py](simulate.py)：`run_cat()` | 初期値を `U(-0.5,0.5)` から生成し、各回答後に `mle(...)` で更新 | 初期値を `np.zeros(n)` にし、各回答後に `eap(...)` で更新。検証・最終評価・選択分析と、各選択規則に渡す能力値を EAP に統一した。 |
| [dqn.py](dqn.py)：`DQNAgent.train()` の初期化 | 初期推定値をランダムに生成し、初期対数分散を0（分散1）に設定 | 初期推定値を0に設定。空の回答記録を `posterior()` に渡して初期分散を求め、その対数を使う。EAP と分散を同じ有限格子の分布にそろえた。 |
| [dqn.py](dqn.py)：`DQNAgent.train()` の回答後更新 | `mle()` で点推定し、別途 `posterior()` で分散を計算 | `posterior()` を1回呼び、事後平均を `theta_hat`、事後分散を `var` に受け取る。学習中の状態と報酬に EAP を反映する。 |
| [dqn.py](dqn.py)：参照能力値の受け渡しと説明 | 全回答後の MLE／Dodd 推定値を参照能力値に使用 | 学習中の `theta_hat` と検証時の `hist[-1]` が EAP になるため、参照能力値も EAP に切り替わる。`fi_ref` などの説明と `episode_return()` のコメントを更新した。 |
| [benchmark_cost.py](benchmark_cost.py)：`history()` | ランダムに生成した回答履歴の能力値を `mle(...)` で推定 | `eap(bank, items, resp)` に置き換え。速度測定で各選択手法へ渡す能力値も EAP にした。 |
| [analyze_selection.py](analyze_selection.py)：`by_step()` | 各ステップで無条件に `np.polyfit(...)` による回帰係数を計算 | `theta_hat_prev.nunique() > 1` の場合だけ計算し、それ以外は `np.nan` にする。全員の初期推定値が0になることに対応した。 |
| [__init__.py](__init__.py)、[run_grid.py](run_grid.py) と推定関連ファイルの説明文 | MLE を使う実験として記載 | CPU・EAP 版であり、`existing` を含む全条件で EAP を使うことを明記した。 |

### 能力推定関数の置き換え

EXP001 では、正誤パターンによる分岐、Dodd 規則、40回の二分法を `mle()` 内で実装していました。EXP003 では、この関数全体を次の関数に置き換えています。

```python
def eap(bank, items, resp):
    if items.shape[1] == 0:
        return np.zeros(items.shape[0])
    return posterior(bank, items, resp)[0]
```

`posterior()` は EXP001 にも存在し、すでに事後平均・事後分散・正規化した重みを返していました。その計算を再利用しています。

### DQN の学習中の更新

EXP001 の回答後の更新は、次の2つの推定処理でした。

```python
theta_hat = mle(self.bank, items[:, : t + 1], resp[:, : t + 1], theta_hat)
log_var = np.log(posterior(self.bank, items[:, : t + 1], resp[:, : t + 1])[1])
```

EXP003 では、同じ事後分布から平均と分散をまとめて取得します。

```python
theta_hat, var, _, _ = posterior(self.bank, items[:, : t + 1], resp[:, : t + 1])
log_var = np.log(var)
```

この箇所では、平均を返す `eap()` と分散を返す `posterior()` を個別に呼ぶと計算が重複するため、`posterior()` の戻り値を直接使っています。
`features()` が受け取る `theta_hat`、回答前後の推定値を使う報酬、エピソード終了時の参照能力値が、この更新を通じて EAP にそろいます。
`_reward()` と `episode_return()` の報酬式は引き継ぎ、式へ渡す推定値を変更しています。

### 初期値と乱数の扱い

学習では `theta_hat = np.zeros(cfg.n_env)` とする直前に、EXP001 と同じ `rng.uniform(-0.5, 0.5, size=cfg.n_env)` を呼び、値を破棄しています。初期値を固定したことによって、後続の探索・反応生成用の乱数の位置がずれることを防ぐためです。

一方、`run_cat()` の評価用の初期値と反応生成は別々の乱数ストリームです。初期値用の乱数生成を削除し、反応生成用の `[seed, 1]` のストリームをそのまま使っています。同じ項目列を固定した場合、EXP001 と同じ反応用乱数で比較できます。実際の適応的な項目列は、推定法の変更によって変わり得ます。

### 引き継いだ処理と EXP003 の出力先

[rules.py](rules.py)、[scenarios.py](scenarios.py)、[make_tables.py](make_tables.py) は、現在の EXP001 と同じコードです。項目選択規準、反応生成モデル、集計表の形式を引き継いでいます。MFI などが受け取る点推定値は、`run_cat()` の変更によって EAP になります。

Q ネットワークの構造、学習人数、最適化手法、ハイパーパラメータ、CPU 実行も EXP001 から引き継いでいます。`run_grid.py` の集計・プロット処理は共通のままです。

実験の選択と保存先は、既存の [experiment_runner.py](../experiment_runner.py) がパッケージ名から決定します。EXP003 を追加したことで `--experiment EXP003` が選択可能になり、`result/EXP003/<grid>/` に保存されます。これに合わせて、空の保存先を保持する `result/EXP003/.gitkeep`、README・Colab の EXP003 の案内、[EAP 用テスト](../tests/test_exp003_eap.py) を追加・更新しています。

## 反応生成と実験グリッド

標準では項目数500、1人のテスト長40問、受験者の真の能力分布は `N(0,1)` です。
識別力 `a` は平均1.2・標準偏差0.25の正規分布から正値になるまで再抽出し、困難度 `b` は `N(0,1)` から生成します。
推定モデルには、生成に使った真の `a,b` を既知として渡します。

| `--grid` | 反応生成モデル | DQN 条件 |
|---|---|---|
| `main` | 2PL（全項目 `c=0`） | `existing`、`proposed` |
| `guess` | 3PL（項目ごとに `c ~ U(0.1,0.3)`） | `proposed` |
| `sensitivity` | 2PL | 報酬7種 × 割引率4種（0、0.5、0.9、1）の28条件 |
| `state` | 2PL | `theta`、`theta_step`、`belief` の3条件 |
| `ablation` | 2PL | 既存・提案の設定間で要因を入れ替える12条件 |

すべてのグリッドで、MFI・FIWL・MPWI・MEPV も比較します。`guess` でも能力推定のモデルは2PLです。
`--rep` は反復番号で、推測の有無は `--grid` で決まります。

実装：[scenarios.py](scenarios.py)、[rules.py](rules.py)、[run_grid.py](run_grid.py)。

## DQN の学習設定

以下は `main` の1反復あたりの設定です。反応記録はシミュレーションで生成します。

| 設定 | `existing` | `proposed` |
|---|---|---|
| 学習用人数（`n_episodes`） | 1,000人 | 20,000人 |
| 同時に処理する人数（`n_env`） | 1人 | 32人 |
| 検証用人数（`n_val`） | 200人 | 500人 |
| 最終評価用人数（`--n-test`） | 2,000人 | 2,000人 |
| 状態 | 能力推定値 | 能力推定値・進行度 `t/L`・対数事後分散 |
| 報酬 | 回答前の推定値で評価した選択項目の情報量（`fi_hat_prev`） | 事後精度の増分 `1/V_after − 1/V_before`（`prec_gain`） |
| 隠れ層のユニット数 | 50、30 | 64、64 |
| 全パラメータの非負制約 | あり | なし |
| 割引率 `gamma` | 0.1 | 0.5 |
| リプレイバッファ容量 | 1,000遷移 | 50,000遷移 |
| ターゲットネットワーク更新間隔 | 40回の重み更新ごと | 500回の重み更新ごと |
| ε-greedy の ε | 0.1で固定 | 1.0から0.05へ減衰 |
| 検証の人数間隔（`eval_every`） | 50人 | 2,000人 |

共通設定は Adam、学習率 `1e-3`、ミニバッチ128遷移、勾配ノルム上限10です。
バッファ容量・ミニバッチは、受験者数ではなく1回答に対応する遷移数です。
提案手法の ε は予定学習人数の前半で線形に減衰し、後半は0.05です。検証は処理バッチの終了時に行うため、32人単位の場合は指定人数の境界を越えてから実施します。

40問の場合、生成する学習回答数は既存手法で40,000件、提案手法で800,000件です。
検証・最終評価の記録は重み更新に使いません。学習中は固定した検証集団における「その条件の報酬の収益」が最大となった重みを採用し、真の能力に対する RMSE は重み選択に使いません。

`sensitivity` の参照能力値は最終回答後の EAPです。
`ablation` では設定の入れ替えにより、条件ごとに学習人数などが異なります。

`--quick` は予定学習人数を200人、検証用人数を100人に短縮します。
実際の人数は `ceil(n_episodes / n_env) × n_env` なので、`main` の既存手法は200人、提案手法は224人です。

## 乱数と反復

基本種は `20260904`、反復 `k` の種は `基本種 + k − 1` です。
項目バンクの `a,b` は反復間で固定し、`guess` の `c` は反復ごとに生成します。
学習・検証・最終評価には分離した乱数系列を使い、最終評価の各手法は同じ受験者と反応用の共通乱数で比較します。

`--rep 3` は反復3を1回実行する指定です。原稿の実験計画では、`main/guess` に反復4〜6、その他のグリッドに反復1〜3を使っていますが、CLI が番号を制限しているわけではありません。

## 実行方法

以下はリポジトリ直下で実行します。

```sh
# uv.lock に従って環境を用意
uv sync --locked

# main の反復4を学習・評価
uv run python run_grid.py --experiment EXP003 --grid main --rep 4

# 保存済みの全反復を集計し、画像を生成
uv run python run_grid.py --experiment EXP003 --grid main --aggregate

# 推測ありの条件
uv run python run_grid.py --experiment EXP003 --grid guess --rep 4

# 動作確認用の結果は別のルートへ保存
uv run python run_grid.py --experiment EXP003 --grid main --rep 4 --quick --out /tmp/2pl-dqn-smoke

# 集計済みの結果から表を標準出力へ生成
uv run python make_tables.py --experiment EXP003 main

# 選択項目の分析と推論時間の測定
uv run python analyze_selection.py --experiment EXP003
uv run python benchmark_cost.py --experiment EXP003
```

選択分析は標準で反復4の種を使い、提案手法を1回学習して MFI・MEPV と比較します。
速度測定では項目数500・2,000・8,000、受験者バッチ1・2,000を使います。DQN は未学習の提案構成を使い、項目選択の推論時間を測ります。

## 出力

標準保存先はリポジトリ内の `result/EXP003/` です。

| パス | 内容 |
|---|---|
| `<grid>/rep<k>.csv` | 条件・ステップ別の RMSE、選択項目の種類数、DQN 更新回数・検証収益など |
| `<grid>/bank_true_rep<k>.csv` | 反応生成に使う項目パラメータ `a,b,c` |
| `<grid>/mean.csv` | 条件・ステップ別の RMSE の平均・標本標準偏差・最小・最大、項目種類数の平均、反復件数 |
| `<grid>/mean.png` | 横軸 `step`、縦軸 `rmse_mean`、条件別の折れ線（標準偏差の帯なし） |
| `selection/` | 選択特性・極端な反応・報酬尺度の CSV と `fig_selection.pdf` |
| `cost/cost.csv` | 項目選択の所要時間 |

`--aggregate` は `rep*.csv` を集計し、`mean.csv` と `mean.png` を上書きします。学習は実行しません。
反復が1つだけの場合、標本標準偏差は `NaN` になります。集計・表生成の表示は標準の40問を前提にしています。

`--out /tmp/2pl-dqn-smoke` の場合は `/tmp/2pl-dqn-smoke/EXP003/<grid>/` に保存します。集計・表生成時にも同じ `--out` を指定します。
ログと表のテキストは標準出力なので、ファイル保存にはリダイレクトが必要です。学習済みモデルのファイル保存は実装していません。

共通の実行入口：[experiment_runner.py](../experiment_runner.py)。全体の説明：[README.md](../README.md)。
