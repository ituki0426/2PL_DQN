import irw
from pathlib import Path


datasets = [
    "criticalperiod_syntax",
    "vocabulary_iq",
    "choi_2026_cmsce_2021_2",
    "psychoneurotic_inventory",
    "christensen_2018_wsssf_5831",
    "choi_2026_cmsce_2019_2",
    "tma",
    "choi_2026_cmsce_2020_1",
]


# 保存先
output_dir = Path("irw_datasets")
output_dir.mkdir(
    parents=True,
    exist_ok=True
)


for i, dataset_name in enumerate(datasets, start=1):

    output_path = output_dir / f"{dataset_name}.csv"

    print(
        f"[{i}/{len(datasets)}] "
        f"{dataset_name} を保存中..."
    )

    try:
        irw.download(
            dataset_name,
            path=str(output_path),
            overwrite=True,
        )

        print(
            f"  Saved: {output_path}"
        )

    except Exception as e:
        print(
            f"  ERROR: {e}"
        )


print("完了")
