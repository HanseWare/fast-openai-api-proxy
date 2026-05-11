import asyncio
import time
import datetime
from typing import Dict, List, Tuple, Optional
from access_store import store as access_store
from usage_worker import enqueue_budget_usage

class BudgetService:
    def __init__(self):
        self.locks: Dict[str, asyncio.Lock] = {}
        # Tracks in-memory cost reservations/usage:
        # cache[entity_type][entity_id][window][window_bucket][model_type] = float
        self.memory_usage: Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, float]]]]] = {}
        self.initialized_entities = set()

    def get_lock(self, entity_key: str) -> asyncio.Lock:
        if entity_key not in self.locks:
            self.locks[entity_key] = asyncio.Lock()
        return self.locks[entity_key]

    def _get_bucket(self, window: str, ts: int) -> str:
        dt = datetime.datetime.fromtimestamp(ts)
        if window == "daily":
            return dt.strftime("%Y-%m-%d")
        return dt.strftime("%Y-%m")

    def _get_usage(self, entity_type: str, entity_id: str, window: str, window_bucket: str, model_type: str) -> float:
        if entity_type not in self.memory_usage: return 0.0
        if entity_id not in self.memory_usage[entity_type]: return 0.0
        if window not in self.memory_usage[entity_type][entity_id]: return 0.0
        if window_bucket not in self.memory_usage[entity_type][entity_id][window]: return 0.0
        return self.memory_usage[entity_type][entity_id][window][window_bucket].get(model_type, 0.0)

    def _add_usage(self, entity_type: str, entity_id: str, window: str, window_bucket: str, model_type: str, amount: float):
        if entity_type not in self.memory_usage: self.memory_usage[entity_type] = {}
        if entity_id not in self.memory_usage[entity_type]: self.memory_usage[entity_type][entity_id] = {}
        if window not in self.memory_usage[entity_type][entity_id]: self.memory_usage[entity_type][entity_id][window] = {}
        if window_bucket not in self.memory_usage[entity_type][entity_id][window]: self.memory_usage[entity_type][entity_id][window][window_bucket] = {}
        
        current = self.memory_usage[entity_type][entity_id][window][window_bucket].get(model_type, 0.0)
        self.memory_usage[entity_type][entity_id][window][window_bucket][model_type] = current + amount

    async def _init_entity(self, entity_type: str, entity_id: str):
        entity_key = f"{entity_type}:{entity_id}"
        if entity_key in self.initialized_entities:
            return
            
        usages = await asyncio.to_thread(access_store.get_all_budget_usage, entity_type, entity_id)
        for u in usages:
            mod_scope = u["scope"] or ""
            self._add_usage(entity_type, entity_id, u["window"], u["window_bucket"], mod_scope, u["cost"])

        self.initialized_entities.add(entity_key)

    async def reserve_budget(self, entities: List[Tuple[str, str]], model_type: str, min_credits: float) -> Tuple[bool, Optional[Tuple[str, str]]]:
        now = int(time.time())
        daily_bucket = self._get_bucket("daily", now)
        monthly_bucket = self._get_bucket("monthly", now)

        for entity_type, entity_id in entities:
            entity_key = f"{entity_type}:{entity_id}"
            
            async with self.get_lock(entity_key):
                await self._init_entity(entity_type, entity_id)
                
                budgets = await asyncio.to_thread(access_store.list_budgets, entity_type, entity_id)
                if not budgets:
                    continue 

                applicable_budgets = []
                for b in budgets:
                    b_scope = b["scope"] or ""
                    if b_scope == "" or b_scope == model_type:
                        applicable_budgets.append(b)
                        
                if not applicable_budgets:
                    continue
                    
                can_afford = True
                for b in applicable_budgets:
                    b_window = b["window"]
                    b_scope = b["scope"] or ""
                    b_bucket = daily_bucket if b_window == "daily" else monthly_bucket
                    
                    used = self._get_usage(entity_type, entity_id, b_window, b_bucket, b_scope)

                    if used + min_credits > b["budget_amount"]:
                        can_afford = False
                        break
                        
                if can_afford:
                    # Deduct (reserve) instantly
                    for b in applicable_budgets:
                        b_window = b["window"]
                        b_scope = b["scope"] or ""
                        b_bucket = daily_bucket if b_window == "daily" else monthly_bucket
                        self._add_usage(entity_type, entity_id, b_window, b_bucket, b_scope, min_credits)

                    return True, (entity_type, entity_id)
                    
        return False, None

    async def resolve_budget(self, entity_type: str, entity_id: str, model_type: str, min_credits: float, actual_cost: float):
        now = int(time.time())
        daily_bucket = self._get_bucket("daily", now)
        monthly_bucket = self._get_bucket("monthly", now)
        entity_key = f"{entity_type}:{entity_id}"
        
        async with self.get_lock(entity_key):
            budgets = await asyncio.to_thread(access_store.list_budgets, entity_type, entity_id)
            applicable_budgets = []
            for b in budgets:
                b_scope = b["scope"] or ""
                if b_scope == "" or b_scope == model_type:
                    applicable_budgets.append(b)
                    
            adjustment = actual_cost - min_credits
            
            for b in applicable_budgets:
                b_window = b["window"]
                b_scope = b["scope"] or ""
                b_bucket = daily_bucket if b_window == "daily" else monthly_bucket
                
                # Update in-memory
                self._add_usage(entity_type, entity_id, b_window, b_bucket, b_scope, adjustment)

                # Push actual cost directly to async worker to persist DB increment
                enqueue_budget_usage(entity_type, entity_id, b_window, b_bucket, actual_cost, b_scope)

budget_service = BudgetService()
