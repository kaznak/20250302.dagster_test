import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from dagster import (AssetExecutionContext, AssetIn, AssetKey, Definitions,
                     DynamicPartitionsDefinition, Output, RunRequest,
                     SensorEvaluationContext, asset, define_asset_job, sensor)
from PIL import Image, ImageEnhance, ImageFilter

# 画像ディレクトリの設定
BASED = Path(__file__).resolve().parents[1]
IMAGE_DIRS = {
    "input": os.path.join(BASED, "data/input_images"),
    "output": os.path.join(BASED, "data/output_images"),
}

for dir in IMAGE_DIRS.values():
    os.makedirs(dir, exist_ok=True)

# 処理タイプ定義
PROCESSING_TYPES = {
    "deterministic": "決定的処理（グレースケール変換）",
    "non_deterministic": "非決定的処理（ランダム効果）",
}

# 動的パーティションの定義
image_partitions = DynamicPartitionsDefinition(name="image_partitions")


# 画像ファイル名の一覧を取得する関数
def get_image_files() -> List[str]:
    """指定されたディレクトリ内の画像ファイルのリストを返す"""
    image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"]
    image_files = []

    for file in os.listdir(IMAGE_DIRS["input"]):
        file_path = os.path.join(IMAGE_DIRS["input"], file)
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
    if not hasattr(context, "partition_key") or context.partition_key is None:
        raise ValueError(
            "このアセットはパーティションキーを指定して実行する必要があります"
        )

    # パーティション（画像ファイル名）を取得
    partition_key = context.partition_key
    image_path = os.path.join(IMAGE_DIRS["input"], partition_key)

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


# ステップ2: 決定的処理（元の処理を維持）
@asset(
    partitions_def=image_partitions,
    key_prefix=["processed_images"],
    ins={
        "image_metadata": AssetIn(key=AssetKey(["registered_images", "register_image"]))
    },
)
def process_image_deterministic(
    context: AssetExecutionContext, image_metadata: Dict[str, Any]
) -> Output[Dict[str, Any]]:
    """登録された画像を決定的に処理するアセット（元のグレースケール変換）"""
    # パーティションキーの存在チェック
    if not hasattr(context, "partition_key") or context.partition_key is None:
        raise ValueError(
            "このアセットはパーティションキーを指定して実行する必要があります"
        )

    # パーティション（画像ファイル名）を取得
    partition_key = context.partition_key
    image_path = image_metadata["image_path"]

    context.log.info(f"画像を決定的に処理中: {image_path}")

    # 画像が存在するか確認
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"画像ファイルが見つかりません: {image_path}")

    # 画像を読み込む
    image = Image.open(image_path)

    # 処理済み画像を保存するディレクトリ
    processed_dir = IMAGE_DIRS["output"]
    processed_path = os.path.join(processed_dir, f"deterministic_{partition_key}")

    # 決定的画像処理を実行（グレースケール変換）
    if image.mode != "L":
        image = image.convert("L")
    image.save(processed_path)

    # 処理結果メタデータ
    processed_metadata = {
        **image_metadata,  # 元のメタデータを継承
        "processed_at": time.time(),
        "processed_path": processed_path,
        "processing_type": "deterministic",
        "processing_description": "grayscale",
        "is_deterministic": True,
    }

    context.log.info(f"決定的画像処理完了: {partition_key} -> {processed_path}")

    return Output(
        processed_metadata,
        metadata={
            "filename": partition_key,
            "original_path": image_path,
            "processed_path": processed_path,
            "processing_type": "deterministic",
        },
    )


# ステップ3: 非決定的処理（新しい処理を追加）
@asset(
    partitions_def=image_partitions,
    key_prefix=["processed_images"],
    ins={
        "image_metadata": AssetIn(key=AssetKey(["registered_images", "register_image"]))
    },
)
def process_image_non_deterministic(
    context: AssetExecutionContext, image_metadata: Dict[str, Any]
) -> Output[Dict[str, Any]]:
    """登録された画像を乱数を使って非決定的に処理するアセット"""
    # パーティションキーの存在チェック
    if not hasattr(context, "partition_key") or context.partition_key is None:
        raise ValueError(
            "このアセットはパーティションキーを指定して実行する必要があります"
        )

    # パーティション（画像ファイル名）を取得
    partition_key = context.partition_key
    image_path = image_metadata["image_path"]

    context.log.info(f"画像を非決定的に処理中: {image_path}")

    # 画像が存在するか確認
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"画像ファイルが見つかりません: {image_path}")

    # 乱数のシード値を生成
    # 現在時刻とパーティションキーからハッシュ値を生成して初期シードとする
    initial_seed = hash(f"{partition_key}_{time.time()}")
    random.seed(initial_seed)

    # 実際に使用する乱数シード値
    seed = random.randint(1, 1000000)

    # シード値を設定
    random.seed(seed)
    np.random.seed(seed)

    context.log.info(f"乱数シード値: {seed}")

    # 画像を読み込む
    image = Image.open(image_path)

    # シンプルに90度、180度、270度のいずれかにランダムで回転させる
    processing_ops = []

    # 回転角度を選択（90度、180度、270度のいずれか）
    rotation_options = [90, 180, 270]
    rotation_angle = random.choice(rotation_options)

    # 画像を回転
    image = image.rotate(rotation_angle, expand=True)
    processing_ops.append(f"回転: {rotation_angle}度")

    # 処理済み画像を保存するディレクトリ
    processed_dir = IMAGE_DIRS["output"]
    processed_path = os.path.join(processed_dir, f"non_deterministic_{partition_key}")

    # 処理した画像を保存
    image.save(processed_path)

    # 処理結果メタデータ
    processed_metadata = {
        **image_metadata,  # 元のメタデータを継承
        "processed_at": time.time(),
        "processed_path": processed_path,
        "processing_type": "non_deterministic",
        "random_seed": seed,  # 使用した乱数シード値を保存
        "processing_operations": processing_ops,  # 適用した処理の一覧
        "is_deterministic": False,  # 非決定的処理であることを明示
    }

    context.log.info(f"非決定的画像処理完了: {partition_key} -> {processed_path}")
    context.log.info(f"適用した処理: {', '.join(processing_ops)}")

    return Output(
        processed_metadata,
        metadata={
            "filename": partition_key,
            "original_path": image_path,
            "processed_path": processed_path,
            "processing_type": "non_deterministic",
            "random_seed": seed,  # メタデータにもシード値を含める
            "processing_operations": ", ".join(
                processing_ops
            ),  # 適用した処理をメタデータに追加
        },
    )


# 画像処理タイプを選択するための設定を取得または既定値を返す関数
def get_processing_type_config(context: SensorEvaluationContext) -> str:
    """
    画像処理タイプの設定を取得する
    現在のジョブタグや環境変数から処理タイプを取得、なければ既定値を返す
    """
    # 環境変数から取得を試みる
    processing_type = os.environ.get("IMAGE_PROCESSING_TYPE")

    # タグなどから設定を取得する方法もあれば追加可能

    # 既定値は両方のタイプを実行（または "deterministic" または "non_deterministic" だけを返すことも可能）
    if not processing_type or processing_type not in [
        "deterministic",
        "non_deterministic",
        "both",
    ]:
        processing_type = "both"

    context.log.info(f"画像処理タイプ設定: {processing_type}")
    return processing_type


# 決定的処理のジョブ定義
deterministic_processing_job = define_asset_job(
    name="deterministic_processing_job",
    selection=[
        AssetKey(["registered_images", "register_image"]),
        AssetKey(["processed_images", "process_image_deterministic"]),
    ],
)

# 非決定的処理のジョブ定義
non_deterministic_processing_job = define_asset_job(
    name="non_deterministic_processing_job",
    selection=[
        AssetKey(["registered_images", "register_image"]),
        AssetKey(["processed_images", "process_image_non_deterministic"]),
    ],
)

# 両方の処理を含むジョブ定義
all_processing_job = define_asset_job(
    name="all_processing_job",
    selection=[
        AssetKey(["registered_images", "register_image"]),
        AssetKey(["processed_images", "process_image_deterministic"]),
        AssetKey(["processed_images", "process_image_non_deterministic"]),
    ],
)


# 新しい画像を検出し、選択された処理タイプで処理するセンサー
@sensor(
    job=all_processing_job,  # 既定では両方のジョブを実行
)
def image_sensor(context: SensorEvaluationContext):
    """
    画像ディレクトリを監視し、新しい画像を検出して処理するセンサー
    処理タイプの設定に基づいて、決定的処理、非決定的処理、または両方を実行
    """
    # 現在のパーティションを取得
    try:
        current_partitions = set(
            context.instance.get_dynamic_partitions(image_partitions.name)
        )
    except Exception as e:
        context.log.error(f"パーティション取得中にエラー: {e}")
        current_partitions = set()

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
    try:
        context.instance.add_dynamic_partitions(
            image_partitions.name, list(new_partitions)
        )
        context.log.info("パーティションを追加しました")
    except Exception as e:
        context.log.error(f"パーティション追加中にエラー: {e}")
        # 代替手段としてDynamicPartitionsDefinitionのAPIを直接使用
        try:
            for partition in new_partitions:
                context.instance.add_dynamic_partition(image_partitions.name, partition)
            context.log.info("個別に各パーティションを追加しました")
        except Exception as e2:
            context.log.error(f"個別パーティション追加中にエラー: {e2}")
            return

    # 処理タイプの設定を取得
    processing_type = get_processing_type_config(context)

    # 登録アセットのキー
    register_asset_key = AssetKey(["registered_images", "register_image"])

    # 処理タイプに応じて処理アセットのキーを選択
    processing_asset_keys = []
    if processing_type in ["deterministic", "both"]:
        processing_asset_keys.append(
            AssetKey(["processed_images", "process_image_deterministic"])
        )

    if processing_type in ["non_deterministic", "both"]:
        processing_asset_keys.append(
            AssetKey(["processed_images", "process_image_non_deterministic"])
        )

    # 各パーティションに対してマテリアライズをリクエスト
    run_requests = []
    for partition in new_partitions:
        run_key_base = f"image_{partition}_{int(time.time())}"

        # まずは登録アセットのマテリアライズをリクエスト
        register_run_key = f"register_{run_key_base}"
        run_requests.append(
            RunRequest(
                run_key=register_run_key,
                asset_selection=[register_asset_key],
                partition_key=partition,
                tags={"partition": partition, "step": "register"},
            )
        )

        # 次に処理アセットのマテリアライズをリクエスト
        for idx, asset_key in enumerate(processing_asset_keys):
            process_run_key = f"process_{idx}_{run_key_base}"
            run_requests.append(
                RunRequest(
                    run_key=process_run_key,
                    asset_selection=[asset_key],
                    partition_key=partition,
                    tags={
                        "partition": partition,
                        "step": "process",
                        "processing_type": asset_key.path[-1],
                    },
                )
            )

    context.log.info(f"{len(new_partitions)}個の新しい画像パーティションを処理します")
    context.log.info(f"処理タイプ: {processing_type}")
    return run_requests


# リソースとジョブの定義
defs = Definitions(
    assets=[
        register_image,
        process_image_deterministic,
        process_image_non_deterministic,
    ],
    sensors=[image_sensor],
    jobs=[
        deterministic_processing_job,
        non_deterministic_processing_job,
        all_processing_job,
    ],
)

if __name__ == "__main__":
    print(
        "Dagsterセンサーベース画像処理パイプライン（決定的/非決定的処理）を起動します..."
    )
    print(f"利用可能な処理タイプ: {PROCESSING_TYPES}")
    print("処理タイプの設定方法: 環境変数 IMAGE_PROCESSING_TYPE を設定してください")
    print("  - deterministic: 決定的処理のみ実行")
    print("  - non_deterministic: 非決定的処理のみ実行")
    print("  - both: 両方の処理を実行（既定値）")
