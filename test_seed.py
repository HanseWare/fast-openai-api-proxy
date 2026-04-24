import os
import json
import sqlite3
import sys
import yaml
import pprint

sys.path.append(os.path.join(os.path.dirname(__file__), "backend", "app"))

from config_store import config_store
from models_handler import ModelsHandler

config_store.db_path = "test_foap.db"
# Clean up old db
if os.path.exists(config_store.db_path):
    os.remove(config_store.db_path)
config_store._init_db()

h = ModelsHandler()

with open("backend/configs/mylab-configs.yaml", "r") as f:
    data = yaml.safe_load(f)
    json_str = data["data"]["mylab-models.json"]
    config = json.loads(json_str)
    
    try:
        h._seed_db_from_json(config)
        print("Success")
    except Exception as e:
        import traceback
        traceback.print_exc()

print("Providers:")
for p in config_store.list_providers():
    print(p["name"])
    for m in config_store.list_models_for_provider(p["id"]):
        print("  " + m["name"])
