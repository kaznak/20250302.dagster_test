import os
import time
from pathlib import Path
from typing import Any, Dict, List

from dagster import (AssetExecutionContext, AssetIn, AssetKey, Definitions,
                     DynamicPartitionsDefinition, Output, RunRequest,
                     ScheduleDefinition, SensorEvaluationContext, asset,
                     define_asset_job, sensor)
from PIL import Image

# 画像ディレクトリの設定
IMAGE_DIR = os.path.join(Path(__file__).parent, "data", "input_images")

# 動的パーティションの定義
image_partitions = DynamicPartitionsDefinition(name="image_partitions")


# 画像ファイル名の一覧を取得する関数
def get_image_files() -> List[str]:
    """指定されたディレクトリ内の画像ファイルのリストを返す"""
    image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"]
    image_files = []

    for file in os.listdir(IMAGE_DIR):
        file_path = os.path.join(IMAGE_DIR, file)
        if os.path.isfile(file_path) and any(
            file.lower().endswith(ext) for ext in image_extensions
        ):
            image_files.append(file)

    return image_files


# ステップ1: 画像を登録するアセット（ファイル自体には何もしない）
@asset(
    partitions_def=image_partitions,
    key_prefix=["registered_images"],
)
def register_image(context: AssetExecutionContext) -> Output[Dict[str, Any]]:
    """パーティション（画像ファイル）を登録するだけのアセット"""
    # パーティションキーの存在チェック
    if not context.has_partition_key:
        raise ValueError(
            "このアセットはパーティションキーを指定して実行する必要があります"
        )

    # パーティション（画像ファイル名）を取得
    partition_key = context.partition_key
    image_path = os.path.join(IMAGE_DIR, partition_key)

    context.log.info(f"画像を登録中: {image_path}")

    # 画像が存在するか確認
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"画像ファイルが見つかりません: {image_path}")

    # 画像のメタデータのみを取得
    image = Image.open(image_path)

    # 登録用メタデータ
    metadata = {
        "filename": partition_key,
        "format": image.format,
        "size": image.size,
        "mode": image.mode,
        "registered_at": time.time(),
        "image_path": image_path,
    }

    context.log.info(f"画像登録完了: {partition_key}")

    return Output(
        metadata,
        metadata={
            "filename": partition_key,
            "original_path": image_path,
        },
    )


# ステップ2: 実際に画像を処理するアセット
@asset(
    partitions_def=image_partitions,
    key_prefix=["processed_images"],
    ins={
        "image_metadata": AssetIn(key=AssetKey(["registered_images", "register_image"]))
    },
)
def process_image(
    context: AssetExecutionContext, image_metadata: Dict[str, Any]
) -> Output[Dict[str, Any]]:
    """登録された画像を実際に処理するアセット"""
    # パーティション（画像ファイル名）を取得
    partition_key = context.partition_key
    image_path = image_metadata["image_path"]

    context.log.info(f"画像を処理中: {image_path}")

    # 画像が存在するか確認（念のため）
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"画像ファイルが見つかりません: {image_path}")

    # 画像を読み込む
    image = Image.open(image_path)

    # 処理済み画像を保存するディレクトリ
    processed_dir = os.path.join(IMAGE_DIR, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    processed_path = os.path.join(processed_dir, f"processed_{partition_key}")

    # 実際の画像処理を実行（例：グレースケール変換）
    if image.mode != "L":
        image = image.convert("L")
    image.save(processed_path)

    # 処理結果メタデータ
    processed_metadata = {
        **image_metadata,  # 元のメタデータを継承
        "processed_at": time.time(),
        "processed_path": processed_path,
        "processing_type": "grayscale",
    }

    context.log.info(f"画像処理完了: {partition_key} -> {processed_path}")

    return Output(
        processed_metadata,
        metadata={
            "filename": partition_key,
            "original_path": image_path,
            "processed_path": processed_path,
        },
    )


# 新しい画像パーティションを検出し、マテリアライズするセンサー
@sensor(
    job_name=None,  # アセットベースのセンサーなのでジョブ名は不要
)
def image_sensor(context: SensorEvaluationContext):
    """
    画像ディレクトリを監視し、新しい画像を検出して処理するセンサー
    1. 新しい画像ファイルを検出
    2. 動的パーティションに追加
    3. 新しいパーティションのマテリアライズをリクエスト
    """
    # 現在のパーティションを取得
    current_partitions = set(image_partitions.get_partitions(context.instance))

    # ディレクトリ内の画像ファイルを取得
    image_files = get_image_files()
    available_partitions = set(image_files)

    # 新しいパーティションを検出
    new_partitions = available_partitions - current_partitions

    if not new_partitions:
        context.log.info("新しい画像パーティションはありません")
        return

    # 新しいパーティションを追加
    context.log.info(f"新しい画像パーティションを追加: {new_partitions}")
    image_partitions.add_partitions(context.instance, list(new_partitions))

    # 新しいパーティションのマテリアライズをリクエスト
    # 登録アセットのキー
    register_asset_key = AssetKey(["registered_images", "register_image"])

    # 各パーティションに対してマテリアライズをリクエスト
    run_requests = []
    for partition in new_partitions:
        run_key = f"register_image_{partition}_{int(time.time())}"

        run_requests.append(
            RunRequest(
                run_key=run_key,
                asset_selection=[register_asset_key],
                partition_key=partition,
                tags={"partition": partition},
            )
        )

    context.log.info(f"{len(new_partitions)}個の新しい画像パーティションを登録します")
    return run_requests


# スケジュールの定義
# 注意: センサーは自動的に実行されるため、通常はスケジュールで実行する必要はありません
# センサー実行用のジョブ定義（アセットでなくセンサーを実行する）
from dagster import sensor_evaluation_job

# センサー評価ジョブの作成
image_sensor_eval_job = sensor_evaluation_job(
    name="image_sensor_eval_job",
    sensors=[image_sensor],  # image_sensorセンサーを実行するジョブ
)

# スケジュール定義
image_sensor_schedule = ScheduleDefinition(
    name="run_image_sensor",
    cron_schedule="*/1 * * * *",  # 1分ごとに実行
    job=image_sensor_eval_job,  # センサー評価ジョブを使用
    execution_timezone="Asia/Tokyo",
)

# リソースとジョブの定義
defs = Definitions(
    assets=[register_image, process_image],
    sensors=[image_sensor],
    schedules=[image_sensor_schedule],
)

if __name__ == "__main__":
    # スクリプトを直接実行したときのために、何か便利な処理を追加することもできます
    print("Dagsterセンサーベース画像処理パイプラインを起動します...")
