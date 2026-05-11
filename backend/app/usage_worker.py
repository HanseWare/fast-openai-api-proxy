import asyncio
import logging
from typing import Optional

from access_store import store as access_store

logger = logging.getLogger(__name__)

usage_queue: asyncio.Queue = asyncio.Queue()

async def usage_worker_task():
    logger.info("Starting background usage logging worker.")
    while True:
        try:
            item = await usage_queue.get()
            if item is None:
                break
                
            task_type = item.get("type")
            if task_type == "log_request":
                await asyncio.to_thread(
                    access_store.log_request,
                    api_key_id=item.get("api_key_id"),
                    timestamp=item.get("timestamp"),
                    model_name=item.get("model_name"),
                    target_model_name=item.get("target_model_name"),
                    provider=item.get("provider"),
                    model_type=item.get("model_type"),
                    usage=item.get("usage"),
                    usage_unit=item.get("usage_unit"),
                    price=item.get("price"),
                    price_per_unit=item.get("price_per_unit"),
                    cost=item.get("cost")
                )
            elif task_type == "add_budget_usage":
                await asyncio.to_thread(
                    access_store.add_budget_usage,
                    entity_type=item.get("entity_type"),
                    entity_id=item.get("entity_id"),
                    window=item.get("window"),
                    window_bucket=item.get("window_bucket"),
                    cost=item.get("cost"),
                    model_type=item.get("model_type")
                )
                
            usage_queue.task_done()
        except asyncio.CancelledError:
            logger.info("Usage worker cancelled.")
            break
        except Exception as e:
            logger.error(f"Error in usage worker: {e}")

def enqueue_request_log(api_key_id: Optional[str], timestamp: int, model_name: str, target_model_name: str,
                        provider: str, model_type: str, usage: float, usage_unit: str, price: float, price_per_unit: float, cost: float):
    usage_queue.put_nowait({
        "type": "log_request",
        "api_key_id": api_key_id,
        "timestamp": timestamp,
        "model_name": model_name,
        "target_model_name": target_model_name,
        "provider": provider,
        "model_type": model_type,
        "usage": usage,
        "usage_unit": usage_unit,
        "price": price,
        "price_per_unit": price_per_unit,
        "cost": cost
    })

def enqueue_budget_usage(entity_type: str, entity_id: str, window: str, window_bucket: str, cost: float, model_type: Optional[str] = None):
    usage_queue.put_nowait({
        "type": "add_budget_usage",
        "entity_type": entity_type,
        "entity_id": entity_id,
        "window": window,
        "window_bucket": window_bucket,
        "cost": cost,
        "model_type": model_type
    })
