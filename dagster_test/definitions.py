import os
import random
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from dagster import (AssetExecutionContext, AssetIn, AssetKey,
                     AssetObservation, Definitions,
                     DynamicPartitionsDefinition, MetadataValue, Output,
                     RunRequest, SensorEvaluationContext, asset,
                     define_asset_job, sensor)
from PIL import Image

# 画像ディレクトリの設定
BASED = Path(__file__).resolve().parents[1]
IMAGE_DIRS = {
    "input": os.path.join(BASED, "data/input_images"),
    "output": os.path.join(BASED, "data/output_images"),
    "versions": os.path.join(BASED, "data/image_versions"),  # バージョン保存用
}

for dir in IMAGE_DIRS.values():
    os.makedirs(dir, exist_ok=True)


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


# ステップ2: 非決定的な処理で画像を処理するアセット
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
    """登録された画像を非決定的に処理するアセット"""
    # パーティションキーの存在チェック
    if not hasattr(context, "partition_key") or context.partition_key is None:
        raise ValueError(
            "このアセットはパーティションキーを指定して実行する必要があります"
        )

    # パーティション（画像ファイル名）を取得
    partition_key = context.partition_key
    image_path = image_metadata["image_path"]

    context.log.info(f"画像を処理中: {image_path}")

    # 画像が存在するか確認（念のため）
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"画像ファイルが見つかりません: {image_path}")

    # バージョンIDを生成（UUID）
    version_id = str(uuid.uuid4())

    # 乱数のシード値を作成
    seed = int(time.time() * 1000) % (2**32)

    # シード値を設定して乱数を初期化
    random.seed(seed)
    np.random.seed(seed)

    # 画像を読み込む
    image = Image.open(image_path)

    # 画像をNumPy配列に変換
    img_array = np.array(image)

    # 非決定的処理: シード値に基づく処理を行う
    # 1. グレースケール変換
    if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
        # カラー画像の場合
        # ランダムな重みでRGBチャンネルを混合
        r_weight = np.random.uniform(0.2, 0.4)
        g_weight = np.random.uniform(0.3, 0.5)
        b_weight = 1.0 - r_weight - g_weight

        # RGB→グレースケール変換（非標準の重み）
        gray_img = (
            img_array[:, :, 0] * r_weight
            + img_array[:, :, 1] * g_weight
            + img_array[:, :, 2] * b_weight
        ).astype(np.uint8)

        processed_img = Image.fromarray(gray_img, mode="L")
    else:
        # すでにグレースケールの場合は、コントラスト調整
        contrast_factor = np.random.uniform(0.8, 1.5)
        processed_img = Image.fromarray(img_array)

        # コントラスト調整
        from PIL import ImageEnhance

        enhancer = ImageEnhance.Contrast(processed_img)
        processed_img = enhancer.enhance(contrast_factor)

    # 2. ノイズ追加（オプション）
    if random.random() > 0.5:
        noise_level = np.random.uniform(5, 15)
        img_array = np.array(processed_img)

        # ノイズ生成
        noise = np.random.normal(0, noise_level, img_array.shape).astype(np.int16)

        # ノイズ追加
        noisy_img = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        processed_img = Image.fromarray(noisy_img)

        if processed_img.mode != "L":
            processed_img = processed_img.convert("L")

    # 処理済み画像を保存するディレクトリ
    processed_dir = IMAGE_DIRS["output"]
    processed_path = os.path.join(processed_dir, f"processed_{partition_key}")

    # バージョン管理用のディレクトリ
    version_dir = os.path.join(IMAGE_DIRS["versions"], partition_key.split(".")[0])
    os.makedirs(version_dir, exist_ok=True)

    # バージョン付きのファイルパス
    versioned_path = os.path.join(version_dir, f"{version_id}_{partition_key}")

    # 画像を保存
    processed_img.save(processed_path)
    processed_img.save(versioned_path)  # バージョン付きでも保存

    # 処理内容の詳細
    processing_details = {
        "seed": seed,
        "version_id": version_id,
    }

    # 追加の処理情報（使用した処理オプション）
    if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
        processing_details.update(
            {
                "r_weight": r_weight,
                "g_weight": g_weight,
                "b_weight": b_weight,
                "type": "custom_grayscale",
            }
        )
    else:
        processing_details.update(
            {
                "contrast_factor": contrast_factor,
                "type": "contrast_adjustment",
            }
        )

    if random.random() > 0.5:
        processing_details.update(
            {
                "noise_added": True,
                "noise_level": noise_level,
            }
        )

    # 処理結果メタデータ
    processed_metadata = {
        **image_metadata,  # 元のメタデータを継承
        "processed_at": time.time(),
        "processed_path": processed_path,
        "versioned_path": versioned_path,
        "version_id": version_id,
        "random_seed": seed,
        "processing_details": processing_details,
    }

    # バージョン情報をオブザベーションとして記録
    context.log_event(
        AssetObservation(
            asset_key=AssetKey(["processed_images", "process_image"]),
            metadata={
                "version": MetadataValue.text(version_id),
                "seed": MetadataValue.int(seed),
                "processing_type": MetadataValue.text(str(processing_details["type"])),
                "partition": MetadataValue.text(partition_key),
            },
        )
    )

    context.log.info(
        f"画像処理完了: {partition_key} -> {processed_path} (version: {version_id})"
    )

    return Output(
        processed_metadata,
        metadata={
            "filename": partition_key,
            "original_path": image_path,
            "processed_path": processed_path,
            "version_id": MetadataValue.text(version_id),
            "random_seed": MetadataValue.int(seed),
            "processing_type": MetadataValue.text(str(processing_details["type"])),
        },
    )


# 過去のバージョンに基づいて画像を再処理するアセット（オプション）
@asset(
    name="reproduce_image",
    key_prefix=["reproduced_images"],
    ins={
        "image_metadata": AssetIn(key=AssetKey(["registered_images", "register_image"]))
    },
)
def reproduce_image(
    context: AssetExecutionContext,
    image_metadata: Dict[str, Any],
    seed: int = None,
    version_id: str = None,
) -> Output[Dict[str, Any]]:
    """過去のバージョンのシード値を使って画像を再処理するアセット"""
    # このアセットはパーティションキーとして画像ファイル名を取得
    partition_key = context.partition_key if hasattr(context, "partition_key") else None

    if partition_key is None:
        raise ValueError("パーティションキーが指定されていません")

    if seed is None and version_id is None:
        raise ValueError("シード値またはバージョンIDを指定する必要があります")

    # バージョンIDが指定されている場合、そのバージョンのメタデータからシード値を取得
    if version_id is not None and seed is None:
        # ここでバージョンに関連するメタデータからシード値を取得する処理を実装
        # 例: DagsterのインスタンスAPIを使用
        # この実装は環境によって異なるため、概念的なコードを示す
        context.log.info(f"バージョンID {version_id} からシード値を検索中...")
        # シード値を取得するロジックをここに実装
        # （実際の実装は複雑になるため概念的に示します）
        seed = context.instance.get_version_metadata(version_id).get("seed")

        if seed is None:
            raise ValueError(f"バージョン {version_id} のシード値が見つかりません")

    # シード値を設定
    context.log.info(f"画像 {partition_key} をシード値 {seed} で再処理します")
    random.seed(seed)
    np.random.seed(seed)

    # 以下、process_imageと同様の処理だが、シード値が固定されている
    image_path = image_metadata["image_path"]

    # 画像を読み込む
    image = Image.open(image_path)

    # 画像をNumPy配列に変換
    img_array = np.array(image)

    # シード値を使った決定的な処理
    if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
        # カラー画像の場合、ランダム（決定的）な重みでRGBチャンネルを混合
        r_weight = np.random.uniform(0.2, 0.4)
        g_weight = np.random.uniform(0.3, 0.5)
        b_weight = 1.0 - r_weight - g_weight

        gray_img = (
            img_array[:, :, 0] * r_weight
            + img_array[:, :, 1] * g_weight
            + img_array[:, :, 2] * b_weight
        ).astype(np.uint8)

        processed_img = Image.fromarray(gray_img, mode="L")
    else:
        # すでにグレースケールの場合、コントラスト調整
        contrast_factor = np.random.uniform(0.8, 1.5)
        processed_img = Image.fromarray(img_array)

        from PIL import ImageEnhance

        enhancer = ImageEnhance.Contrast(processed_img)
        processed_img = enhancer.enhance(contrast_factor)

    # ノイズ追加（オプション）
    if random.random() > 0.5:
        noise_level = np.random.uniform(5, 15)
        img_array = np.array(processed_img)
        noise = np.random.normal(0, noise_level, img_array.shape).astype(np.int16)
        noisy_img = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        processed_img = Image.fromarray(noisy_img)

        if processed_img.mode != "L":
            processed_img = processed_img.convert("L")

    # 再現された画像を保存
    reproduced_dir = os.path.join(IMAGE_DIRS["output"], "reproduced")
    os.makedirs(reproduced_dir, exist_ok=True)

    reproduced_path = os.path.join(reproduced_dir, f"reproduced_{seed}_{partition_key}")
    processed_img.save(reproduced_path)

    # メタデータ
    processed_metadata = {
        **image_metadata,
        "reproduced_at": time.time(),
        "reproduced_path": reproduced_path,
        "original_seed": seed,
        "is_reproduction": True,
    }

    context.log.info(
        f"画像再現完了: {partition_key} -> {reproduced_path} (seed: {seed})"
    )

    return Output(
        processed_metadata,
        metadata={
            "filename": partition_key,
            "original_path": image_path,
            "reproduced_path": reproduced_path,
            "seed": MetadataValue.int(seed),
            "is_reproduction": MetadataValue.bool(True),
        },
    )


# 画像処理パイプラインのジョブ定義
image_processing_job = define_asset_job(
    name="image_processing_job",
    selection=[
        AssetKey(["registered_images", "register_image"]),
        AssetKey(["processed_images", "process_image"]),
    ],
)


# 再現処理用のジョブ定義
reproduction_job = define_asset_job(
    name="reproduction_job",
    selection=[
        AssetKey(["registered_images", "register_image"]),
        AssetKey(["reproduced_images", "reproduce_image"]),
    ],
)


# 新しい画像パーティションを検出し、マテリアライズするセンサー
@sensor(
    job=image_processing_job,  # ジョブを関連付ける
)
def image_sensor(context: SensorEvaluationContext):
    """
    画像ディレクトリを監視し、新しい画像を検出して処理するセンサー
    1. 新しい画像ファイルを検出
    2. 動的パーティションに追加
    3. 新しいパーティションのマテリアライズをリクエスト
    """
    # 現在のパーティションを取得
    try:
        # 新しいAPIを試す
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
        # 新しいAPIを試す
        context.instance.add_dynamic_partitions(
            image_partitions.name, list(new_partitions)
        )
        context.log.info("パーティションを追加しました")
    except Exception as e:
        context.log.error(f"パーティション追加中にエラー: {e}")
        # 代替手段としてDynamicPartitionsDefinitionのAPIを直接使用してみる
        try:
            for partition in new_partitions:
                context.instance.add_dynamic_partition(image_partitions.name, partition)
            context.log.info("個別に各パーティションを追加しました")
        except Exception as e2:
            context.log.error(f"個別パーティション追加中にエラー: {e2}")
            return

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


# リソースとジョブの定義
defs = Definitions(
    assets=[register_image, process_image, reproduce_image],
    sensors=[image_sensor],
    jobs=[image_processing_job, reproduction_job],
)

if __name__ == "__main__":
    print("Dagsterセンサーベース画像処理パイプラインを起動します...")
