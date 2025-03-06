import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from dagster import (AssetExecutionContext, AssetIn, AssetKey, Config,
                     Definitions, Output, RunConfig, RunRequest,
                     SensorEvaluationContext, SensorResult, asset,
                     define_asset_job, sensor)
from PIL import Image

# 画像ディレクトリの設定
BASED = Path(__file__).resolve().parents[1]
IMAGE_DIRS = {
    "input": os.path.join(BASED, "data/input_images"),
    "output": os.path.join(BASED, "data/output_images"),
}

# 画像ディレクトリの作成
for dir in IMAGE_DIRS.values():
    os.makedirs(dir, exist_ok=True)

# 処理タイプ定義
PROCESSING_TYPES = {
    "deterministic": "決定的処理（グレースケール変換）",
    "non_deterministic": "非決定的処理（ランダム効果）",
}


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


class RequestConfig(Config):
    """処理リクエストのRunConfig"""

    request_id: str
    """リクエストID（画像ファイル名）"""


# ステップ1: 画像を登録するアセット（ファイル自体には何もしない）
@asset(
    kinds=["python", "view", "deterministic"],
    group_name="image_test",
    key_prefix=["registered_images"],
    metadata={"deterministic": True},
)
def register_image(
    context: AssetExecutionContext, config: RequestConfig
) -> Output[Dict[str, Any]]:
    """画像ファイルを登録するだけのアセット"""
    # 画像ファイル名を取得
    filename = config.request_id
    image_path = os.path.join(IMAGE_DIRS["input"], filename)

    context.log.info(f"画像を登録中: {image_path}")

    # 画像が存在するか確認
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"画像ファイルが見つかりません: {image_path}")

    # 画像のメタデータのみを取得
    image = Image.open(image_path)

    # 登録用メタデータ
    metadata = {
        "filename": filename,
        "format": image.format,
        "size": image.size,
        "mode": image.mode,
        "registered_at": time.time(),
        "image_path": image_path,
    }

    context.log.info(f"画像登録完了: {filename}")

    return Output(
        metadata,
        metadata={
            "filename": filename,
            "original_path": image_path,
        },
    )


# ステップ2: 決定的処理
@asset(
    kinds=["python", "view", "deterministic"],
    group_name="image_test",
    key_prefix=["processed_images"],
    metadata={"deterministic": True},
    ins={
        "image_metadata": AssetIn(key=AssetKey(["registered_images", "register_image"]))
    },
)
def process_image_deterministic(
    context: AssetExecutionContext,
    image_metadata: Dict[str, Any],
    config: RequestConfig,
) -> Output[Dict[str, Any]]:
    """登録された画像を決定的に処理するアセット（グレースケール変換）"""
    # 画像ファイル名を取得
    filename = config.request_id
    image_path = image_metadata["image_path"]

    context.log.info(f"画像を決定的に処理中: {image_path}")

    # 画像が存在するか確認
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"画像ファイルが見つかりません: {image_path}")

    # 画像を読み込む
    image = Image.open(image_path)

    # 処理済み画像を保存するディレクトリ
    processed_dir = IMAGE_DIRS["output"]
    processed_path = os.path.join(processed_dir, f"deterministic_{filename}")

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

    context.log.info(f"決定的画像処理完了: {filename} -> {processed_path}")

    return Output(
        processed_metadata,
        metadata={
            "filename": filename,
            "original_path": image_path,
            "processed_path": processed_path,
            "processing_type": "deterministic",
        },
    )


# ステップ3: 非決定的処理（バージョン管理機能付き）
@asset(
    kinds=["python", "view"],
    group_name="image_test",
    key_prefix=["processed_images"],
    metadata={"deterministic": False},
    ins={
        "image_metadata": AssetIn(key=AssetKey(["registered_images", "register_image"]))
    },
)
def process_image_non_deterministic(
    context: AssetExecutionContext,
    image_metadata: Dict[str, Any],
    config: RequestConfig,
) -> Output[Dict[str, Any]]:
    """登録された画像を乱数を使って非決定的に処理するアセット（バージョン管理機能付き）"""
    # 画像ファイル名を取得
    filename = config.request_id
    image_path = image_metadata["image_path"]

    context.log.info(f"画像を非決定的に処理中: {image_path}")

    # 画像が存在するか確認
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"画像ファイルが見つかりません: {image_path}")

    # 乱数のシード値を生成
    # 現在時刻とパーティションキーからハッシュ値を生成して初期シードとする
    initial_seed = hash(f"{filename}_{time.time()}")
    random.seed(initial_seed)

    # 実際に使用する乱数シード値
    seed = random.randint(1, 1000000)

    # シード値を設定
    random.seed(seed)
    np.random.seed(seed)

    context.log.info(f"乱数シード値: {seed}")

    # タイムスタンプをバージョン識別子として使用
    timestamp = int(time.time())
    version = f"v{timestamp}"

    # バージョンディレクトリ作成
    version_dir = os.path.join(IMAGE_DIRS["output"], f"versions_{filename}")
    os.makedirs(version_dir, exist_ok=True)

    # 画像を読み込む
    image = Image.open(image_path)

    # シンプルに90度、180度、270度のいずれかにランダムで回転させる
    processing_ops = []

    # 回転角度を選択（90度、180度、270度のいずれか）
    rotation_angle = random.choice([90, 180, 270])

    # 画像を回転
    image = image.rotate(rotation_angle, expand=True)
    processing_ops.append(f"rotate: {rotation_angle} degrees")

    # バージョン付きの保存パス
    version_file_name = f"{version}_{filename}"
    version_path = os.path.join(version_dir, version_file_name)

    # 最新バージョンを示す通常の出力パス
    latest_path = os.path.join(IMAGE_DIRS["output"], f"non_deterministic_{filename}")

    # 処理した画像をバージョン付きで保存
    image.save(version_path)

    # 最新バージョンとしても保存（上書き）
    image.save(latest_path)

    # バージョン情報ファイルの管理（JSON形式）
    version_info_path = os.path.join(version_dir, "version_history.json")
    version_info = {
        "version": version,
        "timestamp": timestamp,
        "seed": seed,
        "operations": processing_ops,
        "image_path": version_path,
    }

    # 既存のバージョン履歴を読み込むか、新規作成
    if os.path.exists(version_info_path):
        try:
            with open(version_info_path, "r") as f:
                version_history = json.load(f)
        except:
            version_history = {"versions": []}
    else:
        version_history = {"versions": []}

    # 新しいバージョン情報を追加
    version_history["versions"].append(version_info)
    version_history["latest_version"] = version

    # バージョン履歴を保存
    with open(version_info_path, "w") as f:
        json.dump(version_history, f, indent=2)

    # 処理結果メタデータ
    processed_metadata = {
        **image_metadata,  # 元のメタデータを継承
        "processed_at": timestamp,
        "processed_path": latest_path,
        "version_path": version_path,
        "version": version,
        "version_history_path": version_info_path,
        "processing_type": "non_deterministic",
        "random_seed": seed,  # 使用した乱数シード値を保存
        "processing_operations": processing_ops,  # 適用した処理の一覧
        "is_deterministic": False,  # 非決定的処理であることを明示
    }

    context.log.info(f"非決定的画像処理完了: {filename}")
    context.log.info(f"バージョン: {version}")
    context.log.info(f"バージョン保存先: {version_path}")
    context.log.info(f"最新バージョン保存先: {latest_path}")
    context.log.info(f"適用した処理: {', '.join(processing_ops)}")

    return Output(
        processed_metadata,
        metadata={
            "filename": filename,
            "original_path": image_path,
            "processed_path": latest_path,
            "version_path": version_path,
            "version": version,
            "processing_type": "non_deterministic",
            "random_seed": seed,
            "processing_operations": ", ".join(processing_ops),
        },
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
    job=all_processing_job,
)
def image_sensor(context: SensorEvaluationContext):
    """
    画像ディレクトリを監視し、新しい画像を検出して処理するセンサー
    決定的処理、非決定的処理の両方を実行
    """
    # 前回のカーソルを取得
    previous_state = json.loads(context.cursor) if context.cursor else {}

    # 各ファイルのマテリアライズをリクエスト
    process_files = []
    run_requests = []
    for file in get_image_files():
        # すでに処理済みのファイルはスキップ
        if file in previous_state:
            continue

        # 新しいファイルの処理
        process_files.append(file)
        previous_state[file] = time.time()

        run_key_base = f"image_{file}_{int(time.time())}"
        request_config = RequestConfig(request_id=file)

        # まずは登録アセットのマテリアライズをリクエスト
        run_key = f"register_{run_key_base}"
        asset_key = AssetKey(["registered_images", "register_image"])
        run_requests.append(
            RunRequest(
                run_key=run_key,
                asset_selection=[asset_key],
                tags={
                    "request_id": file,
                    "step": "register",
                    "processing_type": asset_key.path[-1],
                },
                run_config=RunConfig(
                    ops={
                        asset_key.to_python_identifier(): {
                            "config": request_config._convert_to_config_dictionary()
                        }
                    }
                ),
            )
        )

        # 次に処理アセットのマテリアライズをリクエスト
        for idx, asset_key in enumerate(
            [
                AssetKey(["processed_images", "process_image_deterministic"]),
                AssetKey(["processed_images", "process_image_non_deterministic"]),
            ]
        ):
            run_key = f"process_{idx}_{run_key_base}"
            run_requests.append(
                RunRequest(
                    run_key=run_key,
                    asset_selection=[asset_key],
                    tags={
                        "request_id": file,
                        "step": "process",
                        "processing_type": asset_key.path[-1],
                    },
                    run_config=RunConfig(
                        ops={
                            asset_key.to_python_identifier(): {
                                "config": request_config._convert_to_config_dictionary()
                            }
                        }
                    ),
                )
            )

    context.log.info(f"{len(process_files)}個の新しい画像を処理します")
    return SensorResult(run_requests=run_requests, cursor=json.dumps(previous_state))


dgdef = {
    "assets": [
        register_image,
        process_image_deterministic,
        process_image_non_deterministic,
    ],
    "sensors": [image_sensor],
    "jobs": [
        all_processing_job,
    ],
}


if __name__ == "__main__":
    # リソースとジョブの定義
    defs = Definitions(
        assets=dgdef["assets"],
        sensors=dgdef["sensors"],
        jobs=dgdef["jobs"],
    )
    print(
        "Dagsterセンサーベース画像処理パイプライン（決定的/非決定的処理）を起動します..."
    )
    print(f"利用可能な処理タイプ: {PROCESSING_TYPES}")
    print("処理タイプの設定方法: 環境変数 IMAGE_PROCESSING_TYPE を設定してください")
    print("  - deterministic: 決定的処理のみ実行")
    print("  - non_deterministic: 非決定的処理のみ実行")
    print("  - both: 両方の処理を実行（既定値）")
