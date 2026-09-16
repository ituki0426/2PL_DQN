# EXP005：固定済み項目バンクでの実回答CAT実験

[設計書](design_doc.md)に基づくEXP001由来の独立パッケージ。EXP001のファイルは変更しない。
能力推定はMLE＋Dodd、初期能力値は全員0。CPU・FP32のQネットワークを使用する。

## 入力と実行

対象は `choi_2026_cmsce_2019_2`、`choi_2026_cmsce_2020_1`、`choi_2026_cmsce_2021_2`。
`data/<dataset>/item_parameters.csv`（`item,a,b`）と `person_scores.csv`（`id,theta_EAP`）を読む。
回答は同じディレクトリの `responses.csv`、次に `<dataset>_wide.csv` の順に探す。
別の場所のwide CSVは `--responses` で明示指定する。先頭ID列の名称は `id`。
対象3データセットの回答行列は `irw_datasets/<dataset>_wide.csv` から
`data/<dataset>/responses.csv` へコピー済み。通常の実行では `--responses` は不要。

```sh
# 標準の配置で実行
uv run python run_grid.py --experiment EXP005 --dataset choi_2026_cmsce_2019_2 --rep 1

# 既存のwide CSVを直接指定する場合
uv run python run_grid.py --experiment EXP005 \
  --dataset choi_2026_cmsce_2019_2 \
  --responses irw_datasets/choi_2026_cmsce_2019_2_wide.csv --rep 1

# 感度分析（7報酬 × gamma 4水準、3反復）
for rep in 1 2 3; do
  uv run python run_grid.py --experiment EXP005 \
    --dataset choi_2026_cmsce_2019_2 --grid sensitivity --rep "$rep"
done
uv run python run_grid.py --experiment EXP005 \
  --dataset choi_2026_cmsce_2019_2 --grid sensitivity --aggregate

# データセットごとにDQNを10反復
for dataset in choi_2026_cmsce_2019_2 choi_2026_cmsce_2020_1 choi_2026_cmsce_2021_2; do
  for rep in 1 2 3 4 5 6 7 8 9 10; do
    uv run python run_grid.py --experiment EXP005 --dataset "$dataset" --rep "$rep"
  done
  uv run python run_grid.py --experiment EXP005 --dataset "$dataset" --aggregate
  uv run python make_tables.py --experiment EXP005 --dataset "$dataset" main
done

# 縮小動作確認（40問、学習128人・検証64人・テスト64人、1エポック）
uv run python run_grid.py --experiment EXP005 --dataset choi_2026_cmsce_2019_2 \
  --rep 1 --quick --out /tmp/exp005-smoke
uv run python run_grid.py --experiment EXP005 --dataset choi_2026_cmsce_2019_2 \
  --aggregate --quick --out /tmp/exp005-smoke
```

`--grid` は省略時に `main` となる。`main` では `--conditions existing` または
`--conditions proposed` でDQN条件を限定でき、既存の同じ反復のCSVへ併合する。
`sensitivity` は `prec_gain`、`var_reduction`、`fi_ref`、`fi_hat_prev`、
`fi_hat_post`、`err_reduction_ref`、`neg_sq_err_ref` と割引率0、0.5、0.9、1.0の
全28条件を提案手法の状態・学習設定で比較し、反復1〜3を使う。
`fi_ref`、`err_reduction_ref`、`neg_sq_err_ref` の参照能力には、全項目EAPではなく、
各CATエピソードの全40反応から得た最終MLEを使用してエピソード終了後に報酬を計算する。
`--rep` は1〜10。`--n-epochs` の既定値は両条件とも5、`--test-length` は40。
検証の既定頻度はエポック終了時のみ（`--eval-every 0`）。
検証は必ずエポック終了時に実行し、`--eval-every N` を指定すると累積学習受験者数が
Nの倍数を超えたバッチの終了時にも実行する。同じ時点の検証は重複しない。
`--eval-every 0` はエポック終了時のみ。検証人数は検証分割全員。
`--eval-batch-size`（既定256）は評価時のメモリ使用量を制御する。

## データ整列・分割

IDは文字列として読み、先頭ゼロも保存する。項目・受験者はIDの文字列昇順に整列する。
重複ID・重複列名、ID集合の不一致、欠損、0/1以外の回答、非有限の項目パラメータ・
参照能力値はエラーにする。回答検査は除外前の全項目に行う。
`a <= 0` を除外し、回答行列、項目バンク、出力層とマスクの列順を統一する。
除外後の項目数がテスト長未満なら停止する。

`theta_EAP` の十分位点で層を作り、層内を `split_seed=20260904` で並べ替える。
同値による重複境界はまとめ、全員同値なら1層とする。各層の配分を切り下げた後、
小数部の大きい層から余りを配り、全体の学習人数は `round(N*0.7)`、検証人数は
`round(N*0.1)`、テスト人数は残りとする。全群が非空となる人数が必要。

| データセット | 学習 | 検証 | テスト | 候補項目 | 除外項目 |
|---|---:|---:|---:|---:|---:|
| 2019_2 | 3910 | 559 | 1117 | 290 | 11 |
| 2020_1 | 3772 | 539 | 1077 | 294 | 10 |
| 2021_2 | 4610 | 658 | 1317 | 254 | 11 |

`shuffle_seed=20260904` は学習用乱数から独立で、手法間・反復間で同じ受験者順序を使う。
反復kの学習種は `train_seed + k - 1`（基準値20260904）。初期重み、探索、
リプレイ抽出に使用する。分割用乱数種も反復ごとに `split_seed + k - 1` へ切り替え、
同一反復内では全手法で同じ分割を使い、反復間ではテスト受験者が入れ替わる。
分割用乱数は学習・評価で消費しない。
各エポックで全学習者を1回ずつ使い、端数のバッチも含める。
総エピソード数・総遷移数を結果に記録する。

## 学習・検証・評価

既存条件は `theta` 状態・`fi_hat_prev` 報酬・全パラメータ非負制約・隠れ層50/30、
提案条件は `belief` 状態・`prec_gain` 報酬・制約なし・隠れ層64/64とする。
EXP005では両条件を `n_env=32` とし、同数の実受験者エピソードを同じ並列単位で処理する。
これにより既存条件だけが受験者ごとに勾配更新されることを避け、更新回数と実行時間を
提案条件と同程度に揃える。その他のネットワーク、リプレイと更新設定はEXP001を継承する。
初期事後分散は回答後と同じ−4〜4の81点の格子から計算し、学習・評価・収益計算で揃える。
全項目を使い切る設定でも、終端遷移の将来価値は0として有限の更新値を保つ。

回答は受験者と選択項目に対応する行列要素から取得し、確率生成しない。
参照能力値を環境・Qネットワーク・報酬に渡さない。最良モデルは当該条件の
平均検証エピソード収益が最大の時点とし、同点なら先のモデルを残す。
検証RMSEは診断専用。テスト回答は学習終了後の最終評価にのみ使う。
参照能力値を必要とする3種類の報酬は明示的に拒否する。

評価指標は各ステップのRMSE、MAE、bias、Pearson相関と全テスト履歴で使った項目の種類数。
参照値は全項目の `theta_EAP`。相関は推定値または参照値の分散が0の場合は欠損とする。

## 出力

通常は `result/EXP005/<dataset>/<grid>/`、`--out ROOT` なら
`ROOT/EXP005/<dataset>/<grid>/` に保存する。

| ファイル | 内容 |
|---|---|
| `metadata.json` | 入力のSHA256、設定、種、人数、除外項目、実験の制約 |
| `splits.csv` | 全受験者ID、`theta_reference`、層、所属群、実行での使用有無 |
| `item_bank.csv` | 選別・整列済みの項目IDとa,b |
| `analytic.csv` | 固定テスト群のMFI/FIWL/MPWI/MEPV。`rep=0` は決定論的評価の印 |
| `rep<k>.csv` | DQNの各ステップの指標、種、学習量、最良検証収益と選択時点 |
| `validation_<condition>_rep<k>.csv` | 各検証時点の収益と参照値に対する診断RMSE |
| `model_<condition>_rep<k>.pt` | 選択されたQネットワークの重み、設定、項目順、データ識別情報 |
| `mean.csv`, `mean.png` | 反復の集計とRMSE曲線。`--aggregate` で生成 |

解析的規則も反復ごとにテスト分割が変わるため、DQN条件と同じ反復数で集計し、
実際の `n_rep` と標本SDをそのまま出す。分割は反復間で切り替わるため n_rep は水増しではない。
DQNの集計は存在する反復を使い、実際の `n_rep` を出す。
データ・分割・学習設定が異なる結果を同じ保存先へ混在させる操作は拒否する。
条件を変える場合は別の `--out` を指定する。同じ設定の再実行は対象DQN条件を更新する。
同時実行時の共有結果はLinux/macOSのファイルロックで保護する。

`--quick` は固定分割の各群からID順の先頭だけを使う動作確認で、結果は
`main_quick/` に分離する。実験結果として解釈しない。
`--screening strict` は追加の感度分析 `a >= 0.2` かつ `|b| <= 4` で、
出力は `<dataset>/strict/main/` に分離する。主実験の既定値は正の識別力のみ。

## 解釈上の制約

全受験者（テスト受験者を含む）から推定した項目パラメータを固定して使う予備実験である。
テストデータから完全に独立した評価ではなく、未知集団や未較正項目への汎化性能は示さない。
参照値は全項目回答によるEAP推定値であり、真の能力値ではない。
CATによる提示順序の変更でも回答が変わらないことを仮定する。
これらは出力メタデータ、ログ、集計・表の出力にも明記する。

実データには、正の識別力でも困難度が極端な項目がある。
設計に従い、主実験では除外やDodd推定値の追加クリップを行わない。
全問正答・全問不正答者の推定値はバンク内の困難度最大値・最小値へ近づくため、
MLEの通常の探索範囲を大きく超え、RMSEが大きくなる場合がある。

## 検証

```sh
uv run python -m unittest discover -s tests -p 'test_exp005_real_data.py' -v
```

ID整列、不正入力、層化分割、実回答の使用、MLE＋Doddの一致、項目再使用の拒否、
端数バッチと学習量、全項目消費時の更新、参照値に依存しない学習とモデル選択、
収益の整合、指標、CLI実行、解析的規則のキャッシュ、集計と設定混在の拒否を確認する。
