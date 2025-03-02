import os
from typing import List, Dict, Any, Optional, Set, Callable
from pathlib import Path
import time
import threading
import queue
import logging

# ロガーの設定
logger = logging.getLogger(__name__)

# watchdogのインポート（pyinotifyの代わり）
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileMovedEvent

from dagster import (
    AssetExecutionContext,
    asset,
    AutoMaterializePolicy,
    AssetKey,
    Output,
    DynamicPartitionsDefinition,
    Definitions,
    AssetSelection,
    RunRequest,
    sensor,
    SensorEvaluationContext,
    RunConfig,
    SkipReason,
    ConfigurableResource,
    Config,
    resource,
    Field,
)
from pydantic import Field as PydanticField
from PIL import Image
import numpy as np

# 動的パーティションの定義 - 1画像1パーティション
image_partitions = DynamicPartitionsDefinition(name="image_partitions")

# FileWatcherの設定クラス
class FileWatcherConfig(Config):
    watch_dir: str = PydanticField(description="監視対象のディレクトリ")
    file_extensions: List[str] = PydanticField(
        default=[],
        description="監視対象のファイル拡張子（空の場合は全てのファイルを検出）"
    )

# FileWatcherリソース
class FileWatcherResource(ConfigurableResource):
    config: FileWatcherConfig
    
    def __init__(self):
        self.observer = None
        self.thread = None
        self.running = False
        self.callbacks: List[Callable[[str], None]] = []
        self.new_files_queue = queue.Queue()
    
    def setup(self):
        """リソースの初期化"""
        self.start_watching()
        
    def teardown(self):
        """リソースの終了処理"""
        self.stop_watching()
    
    def register_callback(self, callback: Callable[[str], None]):
        """新しいファイル通知のコールバックを登録"""
        self.callbacks.append(callback)
    
    def _handle_new_file(self, pathname: str):
        """新しいファイルが検出されたときの処理"""
        filename = os.path.basename(pathname)
        
        # 拡張子フィルタリング（file_extensionsが空の場合は全てのファイルを検出）
        if not self.config.file_extensions or any(pathname.lower().endswith(ext) for ext in self.config.file_extensions):
            # キューにファイル名を追加
            self.new_files_queue.put(filename)
            
            # 登録済みコールバックを実行
            for callback in self.callbacks:
                try:
                    callback(filename)
                except Exception as e:
                    logger.error(f"コールバック実行エラー: {e}")
    
    def get_new_files(self) -> List[str]:
        """キューから新しいファイル名の一覧を取得"""
        files = []
        try:
            while not self.new_files_queue.empty():
                files.append(self.new_files_queue.get_nowait())
        except queue.Empty:
            pass
        return files
    
    def start_watching(self):
        """監視を開始"""
        if self.running:
            return
        
        class WatchdogHandler(FileSystemEventHandler):
            def __init__(self, outer):
                self.outer = outer
                
            def on_created(self, event):
                if not event.is_directory:  # ディレクトリではなくファイルの場合
                    self.outer._handle_new_file(event.src_path)
                    
            def on_moved(self, event):
                if not event.is_directory:  # ディレクトリではなくファイルの場合
                    self.outer._handle_new_file(event.dest_path)
        
        # イベントハンドラを設定
        handler = WatchdogHandler(self)
        self.observer = Observer()
        
        # ディレクトリの監視を開始
        watch_path = self.config.watch_dir
        self.observer.schedule(handler, watch_path, recursive=False)
        self.observer.start()
        
        # 監視状態を更新
        self.running = True
        logger.info(f"ディレクトリの監視を開始: {watch_path}")
    
    def stop_watching(self):
        """監視を停止"""
        if not self.running:
            return
        
        self.running = False
        if self.observer:
            self.observer.stop()
            self.observer.join(timeout=5)
        logger.info("ディレクトリの監視を停止")
    
    def get_file_path(self, filename: str) -> str:
        """ファイルの完全パスを取得"""
        return os.path.join(self.config.watch_dir, filename)

# センサーでの実装 - FileWatcherリソースを使用
@sensor(job=None)
def image_detection_sensor(context: SensorEvaluationContext, file_watcher: FileWatcherResource):
    # 新しいファイルを取得
    new_files = file_watcher.get_new_files()
    
    if not new_files:
        return SkipReason("新しい画像はありません")
    
    # 新しい画像パーティションを追加
    context.log.info(f"{len(new_files)}件の新しい画像が検出されました: {new_files}")
    
    current_partitions = set(image_partitions.get_partitions(context.instance))
    added_partitions = []
    
    for image_file in new_files:
        if image_file not in current_partitions:
            image_partitions.add_partitions(context.instance, [image_file])
            added_partitions.append(image_file)
            context.log.info(f"画像パーティションを追加: {image_file}")
    
    if not added_partitions:
        return SkipReason("すべての画像は既にパーティションとして追加されています")
    
    # 新しいパーティションに対するマテリアライゼーションを要求
    return RunRequest(
        run_key=f"process_images_{int(time.time())}",
        asset_selection=AssetSelection.assets(process_image),
        partition_keys=added_partitions,
        run_config=RunConfig(
            ops={"process_image": {"config": {"retry_policy": {"max_retries": 1}}}}
        )
    )

# 画像を処理する外部アセット - 1画像1パーティション
@asset(
    partitions_def=image_partitions,  # 各パーティションは1つの画像ファイルに対応
    auto_materialize_policy=AutoMaterializePolicy.eager(),
    key_prefix=["processed_images"],
)
def process_image(context: AssetExecutionContext, file_watcher: FileWatcherResource) -> Output[Dict[str, Any]]:
    """パーティション（画像ファイル）ごとに画像処理を実行する - 1画像1パーティション"""
    # パーティション（画像ファイル名）を取得
    partition_key = context.partition_key
    image_path = file_watcher.get_file_path(partition_key)
    
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
    processed_dir = os.path.join(file_watcher.config.watch_dir, "processed")
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

# リソースの定義
resources = {
    "file_watcher": FileWatcherResource(
        FileWatcherConfig(
            watch_dir="/path/to/watch/directory",
            # 一般的な画像ファイルの拡張子を設定
            file_extensions=[".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"]
        )
    ),
}

# リソースの初期化と終了処理を行うI/Oマネージャ
defs = Definitions(
    assets=[process_image],
    sensors=[image_detection_sensor],
    resources=resources,
)

if __name__ == "__main__":
    # ロギングの基本設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    logger.info("Dagster watchdog 画像処理パイプラインを起動します...")
    # Dagsterを通して実行する場合、リソースは自動的に初期化・終了されるので
    # ここでは特別な処理は不要