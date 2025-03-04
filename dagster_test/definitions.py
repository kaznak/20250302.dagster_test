from dagster import Definitions

import dagster_test.local_fileprocess_20250304 as lfp20250304

# リソースとジョブの定義
defs = Definitions(
    assets=lfp20250304.dgdef["assets"],
    sensors=lfp20250304.dgdef["sensors"],
    jobs=lfp20250304.dgdef["jobs"],
)

if __name__ == "__main__":
    print(
        "Dagsterセンサーベース画像処理パイプライン（決定的/非決定的処理）を起動します..."
    )
