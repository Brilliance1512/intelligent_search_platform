import json
import secrets
import string
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import redis

from app.config import DIALOG_HISTORY_LIMIT, DIALOG_TTL_SECONDS, REDIS_URL, SCHEMA_CACHE_TTL_SECONDS


@dataclass
class SchemaCacheRecord:
    tenant_id: str
    dataset_hash: str
    config: dict[str, Any]
    schema: str
    created_at: str


class RedisStore:
    def __init__(self, url: str = REDIS_URL):
        self.client = redis.Redis.from_url(url, decode_responses=True)

    def ping(self) -> bool:
        return bool(self.client.ping())

    def key(self, *parts: str) -> str:
        return ":".join(p.replace(":", "_") for p in parts)

    def create_tenant_id(self, length: int = 32) -> str:
        alphabet = string.ascii_letters + string.digits
        while True:
            tenant_id = "".join(secrets.choice(alphabet) for _ in range(length))
            if self.client.sadd("tenants", tenant_id):
                return tenant_id

    def register_tenant(self, tenant_id: str) -> None:
        self.client.sadd("tenants", tenant_id)

    def next_message_id(self, tenant_id: str) -> int:
        self.register_tenant(tenant_id)
        return int(self.client.incr(self.key("tenant", tenant_id, "message_seq")))

    def get_schema(self, tenant_id: str, dataset_hash: str) -> SchemaCacheRecord | None:
        raw = self.client.get(self.key("tenant", tenant_id, "schema", dataset_hash))
        return SchemaCacheRecord(**json.loads(raw)) if raw else None

    def set_schema(self, tenant_id: str, dataset_hash: str, config: dict, schema: str) -> None:
        record = SchemaCacheRecord(tenant_id, dataset_hash, config, schema, datetime.now(timezone.utc).isoformat())
        self.client.setex(
            self.key("tenant", tenant_id, "schema", dataset_hash),
            SCHEMA_CACHE_TTL_SECONDS,
            json.dumps(asdict(record), ensure_ascii=False),
        )

    def get_dataset_meta(self, tenant_id: str, dataset_id: str) -> dict | None:
        raw = self.client.get(self.key("tenant", tenant_id, "dataset", dataset_id, "meta"))
        return json.loads(raw) if raw else None

    def set_dataset_meta(self, tenant_id: str, dataset_id: str, meta: dict) -> None:
        self.client.set(self.key("tenant", tenant_id, "dataset", dataset_id, "meta"), json.dumps(meta, ensure_ascii=False))

    def delete_dataset_meta(self, tenant_id: str, dataset_id: str) -> None:
        self.client.delete(self.key("tenant", tenant_id, "dataset", dataset_id, "meta"))

    def append_dialog_event(self, tenant_id: str, dialog_id: str, event: dict, message_id: int) -> None:
        key = self.key("tenant", tenant_id, "dialog", dialog_id, "events")
        event = {"message_id": message_id, "ts": datetime.now(timezone.utc).isoformat(), **event}
        pipe = self.client.pipeline()
        pipe.rpush(key, json.dumps(event, ensure_ascii=False))
        pipe.ltrim(key, -DIALOG_HISTORY_LIMIT, -1)
        pipe.expire(key, DIALOG_TTL_SECONDS)
        pipe.execute()

    def get_dialog_history(self, tenant_id: str, dialog_id: str, limit: int = DIALOG_HISTORY_LIMIT) -> list[dict]:
        return [json.loads(x) for x in self.client.lrange(self.key("tenant", tenant_id, "dialog", dialog_id, "events"), -limit, -1)]

    def clear_dialog(self, tenant_id: str, dialog_id: str) -> None:
        self.client.delete(self.key("tenant", tenant_id, "dialog", dialog_id, "events"))
