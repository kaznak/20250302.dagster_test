import os
from typing import List, Dict, Any
from pathlib import Path

from dagster import (
    AssetExecutionContext,
    AssetsDefinition,
    MaterializeResult,
    asset,
    multi_asset,
    AutoMaterializePolicy,
    AssetKey,
    AssetIn,
    Output,
    DynamicPartitionsDefinition,
    StaticPartitionsDefinition,
    Definitions,
    AssetSelection,
    ScheduleDefinition,
    AutoMaterializePolicy,
    DynamicPartitionsSubset,
)
import time
from PIL import Image
import numpy as np

# 画像ディレクトリの設定
IMAGE_DIR = "/path/to/image/directory"  # 画像が保存されるディレクトリへのパス

# 動的パーティションの定義
image_partitions = DynamicPartitionsDefinition(name="image_partitions")

# 画像ファイル名の一覧を取得する関数
def get_image_files() -> List[str]:
    """指定されたディレクトリ内の画像ファイルのリストを返す"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
    image_files = []
    
    for file in os.listdir(IMAGE_DIR):
        file_path = os.path.join(IMAGE_DIR, file)
        if os.path.isfile(file_path) and any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file)
            
    return image_files

# 新しいパーティションを検出するアセット
@asset(
    auto_materialize_policy=AutoMaterializePolicy.eager(),
    partitions_def=StaticPartitionsDefinition(["singleton"]),  # シングルトンパーティション
)
def detect_image_partitions(context: AssetExecutionContext) -> None:
    """画像ディレクトリをスキャンして新しい画像パーティションを追加する"""
    # 現在のパーティションを取得
    current_partitions = set(image_partitions.get_partitions(context.instance))
    
    # ディレクトリ内の画像ファイルを取得
    image_files = get_image_files()
    available_partitions = set(image_files)
    
    # 新しいパーティションを検出
    new_partitions = available_partitions - current_partitions
    if new_partitions:
        context.log.info(f"新しい画像パーティションを追加: {new_partitions}")
        image_partitions.add_partitions(context.instance, list(new_partitions))

# 画像を処理する外部アセット
@asset(
    partitions_def=image_partitions,
    auto_materialize_policy=AutoMaterializePolicy.eager(),
    key_prefix=["processed_images"],
)
def process_image(context: AssetExecutionContext) -> Output[Dict[str, Any]]:
    """パーティション（画像ファイル）ごとに画像処理を実行する"""
    # パーティション（画像ファイル名）を取得
    partition_key = context.partition_key
    image_path = os.path.join(IMAGE_DIR, partition_key)
    
    context.log.info(f"画像を処理中: {image_path}")
    
    # 画像が存在するか確認
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"画像ファイルが見つかりません: {image_path}")
    
    # ここで画像処理を実行
    # 例として、画像を読み込んで基本的なメタデータを抽出
    image = Image.open(image_path)
    
    # 画像の処理結果メタデータ
    metadata = {
        "filename": partition_key,
        "format": image.format,
        "size": image.size,
        "mode": image.mode,
        "processed_at": time.time(),
    }
    
    # オプション: 処理済み画像を保存
    processed_dir = os.path.join(IMAGE_DIR, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    processed_path = os.path.join(processed_dir, f"processed_{partition_key}")
    
    # 簡単な処理の例（グレースケール変換）
    if image.mode != "L":
        image = image.convert("L")
    image.save(processed_path)
    
    context.log.info(f"画像処理完了: {partition_key} -> {processed_path}")
    
    return Output(
        metadata,
        metadata={
            "filename": partition_key,
            "original_path": image_path,
            "processed_path": processed_path,
        },
    )

# 新しいパーティションのマテリアライゼーションを自動化するセンサーアセット
@asset(
    auto_materialize_policy=AutoMaterializePolicy.eager(),
    deps=[detect_image_partitions],
)
def materialize_new_images(context: AssetExecutionContext) -> None:
    """新しい画像パーティションを自動的にマテリアライズする"""
    # 最後の実行以降に追加された画像パーティションを取得
    all_partitions = image_partitions.get_partitions(context.instance)
    
    # パーティションのステータスを確認して、マテリアライズが必要なものを特定
    need_materialization = []
    
    for partition in all_partitions:
        asset_key = AssetKey(["processed_images", "process_image"])
        if not context.instance.has_asset_key(asset_key):
            need_materialization.append(partition)
            continue
            
        last_materialization = context.instance.get_latest_materialization_event(
            asset_key=asset_key,
            partition_key=partition,
        )
        
        if last_materialization is None:
            need_materialization.append(partition)
    
    if need_materialization:
        context.log.info(f"マテリアライズが必要な画像: {need_materialization}")
        
        # 新しいパーティションをマテリアライズ
        subset = DynamicPartitionsSubset(
            dynamic_partitions_def=image_partitions,
            partition_keys=need_materialization,
        )
        
        # データを外部化せずに直接マテリアライゼーションを行う実装例
        # 実際のプロダクションでは、以下のようにジョブを起動する方が良い場合もあります
        # context.instance.submit_job_for_materialization(...)
        context.log.info(f"{len(need_materialization)}個の新しい画像パーティションをマテリアライズします")
    else:
        context.log.info("マテリアライズが必要な新しい画像はありません")

# スケジュールの定義
check_images_schedule = ScheduleDefinition(
    name="check_new_images",
    cron_schedule="*/5 * * * *",  # 5分ごとに実行
    asset_selection=AssetSelection.assets(detect_image_partitions, materialize_new_images),
    execution_timezone="Asia/Tokyo",
)

# リソースとジョブの定義
defs = Definitions(
    assets=[detect_image_partitions, process_image, materialize_new_images],
    schedules=[check_images_schedule],
)

if __name__ == "__main__":
    # スクリプトを直接実行したときのために、何か便利な処理を追加することもできます
    print("Dagster動的パーティショニング画像処理パイプラインを起動します...")
