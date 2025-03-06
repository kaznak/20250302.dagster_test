import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List

from dagster import (AssetExecutionContext, AssetIn, AssetKey, Config,
                     ConfigurableIOManager, Definitions, InputContext,
                     IOManager, OutputContext, RunConfig, RunRequest,
                     SensorEvaluationContext, SensorResult, asset,
                     define_asset_job, sensor)
from PIL import Image

BASED = Path(__file__).resolve().parents[1]
IMAGE_DIR = os.path.join(BASED, "data/image_processor")


class ImageDirectoryIOManager(ConfigurableIOManager):
    def create_io_manager(self, context) -> IOManager:
        os.makedirs(IMAGE_DIR, exist_ok=True)
        return self

    def load_input(self, context: InputContext) -> Image:
        filename = os.path.join(self.IMAGE_DIR, context.tags["request_id"])
        return Image.open(filename)

    def handle_output(self, context: OutputContext, image: Image):
        request_id = os.path.join(self.IMAGE_DIR, context.tags["request_id"])
        filename = os.path.join(IMAGE_DIR, f"{context.asset_key.path}_{request_id}")
        image.save(filename)


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
def registered_image(
    context: AssetExecutionContext, config: RequestConfig, image: Image
) -> Image:
    """画像ファイルを登録するだけのアセット"""
    context.add_asset_metadata(
        {
            "filename": config.request_id,
            "format": image.format,
            "size": image.size,
            "mode": image.mode,
            "registered_at": time.time(),
        }
    )
    return image


# ステップ2: 決定的処理
@asset(
    kinds=["python", "view", "deterministic"],
    group_name="image_test",
    key_prefix=["processed_images"],
    metadata={"description": "画像をグレースケール変換", "deterministic": True},
    ins={
        "registered_image": AssetIn(
            key=AssetKey(["registered_images", "registered_image"])
        )
    },
)
def deterministic_processed_image(
    context: AssetExecutionContext,
    registered_image: Image,
) -> Image:
    """登録された画像を決定的に処理するアセット（グレースケール変換）"""
    if registered_image.mode != "L":
        registered_image = registered_image.convert("L")
    context.add_asset_metadata(
        {
            "processed_at": time.time(),
        }
    )
    return registered_image


# ステップ3: 非決定的処理（バージョン管理機能付き）
@asset(
    kinds=["python", "view"],
    group_name="image_test",
    key_prefix=["processed_images"],
    metadata={"deterministic": False},
    ins={
        "registered_image": AssetIn(
            key=AssetKey(["registered_images", "registered_image"])
        )
    },
)
def non_deterministic_processed_image(
    context: AssetExecutionContext,
    registered_image: Image,
) -> Image:
    """登録された画像を乱数を使って非決定的に処理するアセット（バージョン管理機能付き）"""
    stime = time.time()

    def random_seed(context=context, stime=stime):
        """乱数シード値を生成する関数"""
        # 現在時刻とファイル名からハッシュ値を生成して初期シードとする
        filename = context.metadata["filename"]
        initial_seed = hash(f"{filename}_{stime}")
        random.seed(initial_seed)
        # 実際に使用する乱数シード値
        seed = random.randint(1, 1000000)

    # シード値を設定
    seed = random_seed()
    random.seed(seed)
    context.log.info(f"乱数シード値: {seed}")

    # シンプルに90度、180度、270度のいずれかにランダムで回転させる
    processing_ops = []

    # 回転角度を選択（90度、180度、270度のいずれか）
    rotation_angle = random.choice([90, 180, 270])

    # 画像を回転
    registered_image = registered_image.rotate(rotation_angle, expand=True)
    processing_ops.append(f"rotate: {rotation_angle} degrees")

    # 処理結果メタデータ
    context.add_asset_metadata(
        {
            "processed_at": stime,
            "random_seed": seed,  # 使用した乱数シード値を保存
            "processing_operations": processing_ops,  # 適用した処理の一覧
        }
    )

    return registered_image


# 両方の処理を含むジョブ定義
all_processing_job = define_asset_job(
    name="all_processing_job",
    selection=[
        AssetKey(["registered_images", "registered_image"]),
        AssetKey(["processed_images", "deterministic_processed_image"]),
        AssetKey(["processed_images", "non_deterministic_processed_image"]),
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

    # 画像ファイル名の一覧を取得する関数
    def get_files(
        prefix="input_", extentions=[".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"]
    ) -> List[str]:
        """指定されたディレクトリ内の画像ファイルのリストを返す"""
        files = []

        for file in os.listdir(IMAGE_DIR):
            file_path = os.path.join(IMAGE_DIR, file)
            if (
                os.path.isfile(file_path)
                and any(file.lower().endswith(ext) for ext in extentions)
                and file.startswith(prefix)
            ):
                files.append(file)

        return files

    # 各ファイルのマテリアライズをリクエスト
    process_files = []
    run_requests = []
    for file in get_files():
        # すでに処理済みのファイルはスキップ
        if file in previous_state:
            continue

        # 新しいファイルの処理
        process_files.append(file)
        previous_state[file] = time.time()

        run_key_base = f"image_{file}_{int(time.time())}"
        request_config = RequestConfig(request_id=file)

        # アセットのマテリアライズをリクエスト
        for idx, (step, asset_key) in enumerate(
            [
                ("register", AssetKey(["registered_images", "register_image"])),
                (
                    "process",
                    AssetKey(["processed_images", "deterministic_processed_image("]),
                ),
                (
                    "process",
                    AssetKey(["processed_images", "non_deterministic_processed_image"]),
                ),
            ]
        ):
            run_key = f"{step}_{idx}_{run_key_base}"
            run_requests.append(
                RunRequest(
                    run_key=run_key,
                    asset_selection=[asset_key],
                    tags={
                        "request_id": file,
                        "step": step,
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

    if process_files:
        context.log.info(f"{len(process_files)}個の新しい画像を処理します")
    else:
        context.log.debug("新しい画像はありません")
    return SensorResult(run_requests=run_requests, cursor=json.dumps(previous_state))


dgdef = {
    "assets": [
        registered_image,
        deterministic_processed_image,
        non_deterministic_processed_image,
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
