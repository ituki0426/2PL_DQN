# CAT/Experiments: 推定値の不確実性を考慮した状態と報酬に基づく深層強化学習型 CAT の項目選択（実装・実験）

最終更新: 2026-09-08（原稿 v8 の本文で用いる実験だけにコード・結果・ログを整理した）

## この作業領域は何か

Wang, Liu, & Xu (2024, BRM) の DQN による CAT 項目選択の問題点（Q 関数の全パラメータの非負制約，能力推定値だけの状態，
MFI の項目選択規準と一致する報酬）を整理し，最尤推定値・テストの進行度・事後分散からなる状態と，事後精度の増分という
推定値の不確実性を考慮した報酬に基づく，制約なしの DQN を提案・検証する．学習・検証・Q 関数の選択はいずれも反応記録と推定モデルだけから行う．
推定モデルがデータ生成モデルと一致する場合には事後分布に基づく解析的な項目選択規則（MPWI，MEPV）と同等の精度を数値積分なしに達成し，
データ生成モデルが推測を含む 3PL で推定モデルが 2PLM である場合には実際の反応から学習した提案手法が解析的な項目選択規則を小さいながら一貫して上回ることを示す．
原稿は `../Paper/main.tex`（v8，15 頁）．原稿の付録 1・2（削除した第 2・3 段階の検討）の転記元は `../Paper/appendix_sources/` であり，本フォルダのコードとは独立である．

## 構成

```
CAT/
├── README.md                索引
├── Experiments/             本フォルダ
│   ├── README.md, requirements.txt, .venv/（python3 -m venv，uv 不使用）
│   ├── catrl/
│   │   ├── irt.py          2PLM（3PL の反応生成も可），有界二分法の MLE＋Dodd 規則，格子事後分布（G=81，[−4,4]）
│   │   ├── scenarios.py    データ生成モデル: base（2PLM）／guess（c〜U(0.1,0.3) の 3PL）．推定モデルは常に真の a,b の 2PLM
│   │   ├── simulate.py     受験者集団に対する CAT の一括実行（共通乱数；データ生成モデルと推定モデルを分離）と RMSE
│   │   ├── rules.py        MFI，FIWL，MPWI，MEPV
│   │   └── dqn.py          DQN（状態 A/B/C，報酬 7 種，非負制約の有無，検証得点＝当該報酬の収益による Q 関数の選択）
│   ├── run_grid.py         条件グリッドの実行と集計（main / sensitivity / state / ablation / guess → results/<grid>/）
│   ├── make_tables.py      原稿の表の行と対応差（main / sensitivity / state / ablation / guess / cost）
│   ├── analyze_selection.py  選択項目の特性と報酬の尺度（results/selection/，図は ../Paper/fig_selection.pdf）
│   ├── benchmark_cost.py     選択時間（results/cost/，単一スレッド）
│   └── results/<grid>/     rep<k>.csv（反復ごとの各ステップの RMSE，項目数，参照値，更新回数，検証得点），mean.csv，
│                            aggregate.txt，table_rows.txt，bank_true_rep<k>.csv（データ生成モデルの a,b,c），log_rep<k>.txt
├── Paper/                   原稿（IEICE 論文誌 D 体裁）．正本は main.tex．appendix_sources/ に第 2・3 段階の写し
└── related_papers/          文献 PDF
```

## 原稿の節と実験の対応

| 節 | 内容 | 実行 | 条件 | 反復 | 転記元 |
|---|---|---|---|---|---|
| 6.2 主結果（表 2） | 推定モデル＝データ生成モデル（2PLM） | `run_grid.py --grid main` | MFI, FIWL, MPWI, MEPV, existing（既存）, proposed（提案） | 4-6（確証） | `results/main/table_rows.txt` |
| 6.3 感度分析（表 3） | 報酬 7 種 × γ∈{0,0.5,0.9,1}（提案の状態・学習設定） | `--grid sensitivity` | `<報酬>_g<γ>` の 28 条件＋4 規則 | 1-3（探索） | `results/sensitivity/table_rows.txt` |
| 6.4 状態表現（表 4） | 状態 A/B/C（提案の報酬・学習設定） | `--grid state` | state_theta（A）, state_theta_step（B）, state_belief（C）＋4 規則 | 1-3（探索） | `results/state/table_rows.txt` |
| 6.5 要因分解（表 5） | 既存⇄提案の一要因入替 | `--grid ablation` | existing, existing+<要因>×5, proposed, proposed-<要因>×5＋4 規則 | 1-3（探索） | `results/ablation/table_rows.txt` |
| 6.6 選択分析（図 1） | 選択項目の特性，極端な反応パターン，報酬の尺度 | `analyze_selection.py` | MFI, MEPV, 提案（反復 4 の種で 1 回学習） | 4 | `results/selection/log.txt`，`*.csv` |
| 6.7 推定モデル≠データ生成モデル（表 6） | データ生成 3PL（c〜U(0.1,0.3)），推定モデルは真の a,b の 2PLM．提案は 3PL の反応で学習 | `--grid guess` | MFI, FIWL, MPWI, MEPV, proposed | 4-6（確証） | `results/guess/table_rows.txt` |
| 6.8 選択時間（表 7） | 1 受験者・1 ステップの選択時間（I=500/2000/8000，1 人ずつ／2000 人一括） | `benchmark_cost.py` | MFI, FIWL, MPWI, MEPV, DQN (proposed) | — | `results/cost/table_rows.txt` |

コード上の名称と原稿の対応: 状態 `theta`／`theta_step`／`belief`＝状態 A／B／C（C が式 (14)）．報酬 `prec_gain`＝式 (15)（提案），
`var_reduction`＝対数分散減少，`fi_hat_prev`＝式 (11)（既存），`fi_hat_post`＝反応後推定値 FI，`fi_ref`／`err_reduction_ref`／`neg_sq_err_ref`＝参照能力値に基づく 3 報酬（表 3 のみ．エピソード終了後に割当）．
要因分解の要因: constraint（非負制約），state（状態表現），reward（報酬），gamma（割引率），learning（学習設定群）．
既存条件・提案条件の設定値は `run_grid.py` の `EXISTING`・`PROPOSED`（原稿 6.1 節と同じ値）．

## 実行

```sh
cd CAT/Experiments
# 探索的実験（反復1-3）
for g in sensitivity state ablation; do for k in 1 2 3; do .venv/bin/python -u run_grid.py --grid $g --rep $k --threads 2 > results/$g/log_rep$k.txt 2>&1 & done; done; wait
# 確証的実験（反復4-6）
for g in main guess; do for k in 4 5 6; do .venv/bin/python -u run_grid.py --grid $g --rep $k --threads 2 > results/$g/log_rep$k.txt 2>&1 & done; done; wait
# 集計と原稿の表の行
for g in main sensitivity state ablation guess; do .venv/bin/python run_grid.py --grid $g --aggregate > results/$g/aggregate.txt; .venv/bin/python make_tables.py $g > results/$g/table_rows.txt; done
.venv/bin/python analyze_selection.py > results/selection/log.txt   # 反復4の種で提案手法を1回学習し，図を ../Paper/ へ複製
.venv/bin/python benchmark_cost.py > results/cost/log.txt; .venv/bin/python make_tables.py cost > results/cost/table_rows.txt   # 他の処理が走っていないときに
# 一部の DQN 条件だけを再実行して rep csv に併合（解析的な規則は csv がないときだけ再実行）
.venv/bin/python run_grid.py --grid sensitivity --rep 1 --conditions fi_ref_g0.0,fi_ref_g0.5
# 動作確認（各条件を数百エピソードだけ学習．--out を指定して記録済みの結果を上書きしない）
.venv/bin/python run_grid.py --grid main --rep 4 --quick --out /tmp/smoke
```

1 条件の学習は提案側の設定で約 1 分，既存側の設定で約 1 分（Apple M 系 CPU，2 スレッド）．`guess` は 3PL の反応生成のため約 2 分．

## 乱数と反復

- バンクは `(基本種 20260904, 99)` で固定．反復 k の種は 20260904+k−1．推測パラメータは `(種, 30)`，評価受験者は `(種, 20)`，
  評価の初期値と反応は `(種+100, 0/1)`，DQN の学習・検証は `(種, 10/11)` と `種*7+12`．
- `scenarios.py` は推測パラメータの前に長さ I の正規乱数を 2 回消費する（旧版で項目パラメータの誤差に使っていた乱数．記録済みの結果を同じ種で再現するために残している）．
- 反復 1-3 は設計の選択（探索），反復 4-6 は評価（確証）に使い分ける．
- Q 関数の選択は検証集団における当該報酬の収益（提案では 1/V_L − 1/V_0 の平均）．真の能力は使わない．

## 結果の要点（ステップ 40 の RMSE，3 反復平均．正式な転記は `../Paper/main.tex`）

| 設定 | MFI | MEPV | 提案 | 備考 |
|---|---:|---:|---:|---|
| 2PLM が真（`main`，反復 4-6） | 0.218 | 0.220 | 0.220 | 提案はステップ 5 で MFI を 3 反復すべて下回る（−0.017）．MEPV と同等（差 ≤0.004，符号は混在） |
| 推測あり（`guess`，反復 4-6） | 0.492 | 0.471 | 0.468 | 提案−MEPV s40 −0.003 [−0.004,−0.0001] 3/3 負，s10 −0.018 3/3 負．提案−MFI s40 −0.024 3/3 負 |

## 整理の記録

- 2026-09-07: 原稿本体で使わない実験（項目パラメータの再推定を伴う規則，参照オラクル，真値検証，項目パラメータ誤差の掃引，bias・MAE）のコード・結果・HANDOFF を削除．
  `results/misspec/calib0_guess/` を `results/guess/`（`--grid guess`）に改名．
- 2026-09-08: MEPG（本研究で定義した規則），仮定モデルで学習した対照条件，提案の参照報酬版（`proposed_fi_ref`）を削除し，参照能力値に基づく報酬は `sensitivity` にのみ残した．
  `analyze_selection.py` の比較対象を MEPG から MEPV に変えて再実行．rep csv，`log_rep<k>.txt`，`results/cost/{cost.csv,log.txt}` から削除した条件の行を除いた．学習済みモデルは保存しない（乱数種から再現できる）．
- 確認（2026-09-08）: `main` 反復 4 を現在のコードで完全に再実行し，記録済みの rep4.csv と全行一致（RMSE，項目数，更新回数，検証得点）．
  全グリッドの `--quick` 実行，`make_tables.py` の出力（表 2〜7 の全行）が原稿と一致すること．
