"""Card semantic 单源读取：源码扫描生成的 jsonl + sqlite 轻量索引。"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import json
import sqlite3
from functools import lru_cache

from constants import GAME_SEMANTIC_INDEX_DB
from networkV2.s1_schema.card_tags import FUNCTIONAL_TAG_TO_IDX, FUNCTIONAL_TAGS
from networkV2.s1_schema.slug_utils import _slugify


@dataclass(frozen=True)
class CardSemanticSnapshot:
    card_id: str = ""
    class_name: str = ""
    card_type: str = ""
    rarity: str = ""
    target_type: str = ""
    base_cost: int = 0
    is_x_cost: bool = False
    gains_block: bool = False
    db_tags: tuple[str, ...] = ()
    db_keywords: tuple[str, ...] = ()
    functional_tags: tuple[str, ...] = ()
    functional_tag_vec: tuple[float, ...] = ()
    source_path: str = ""

    def has_functional_tag(self, tag: str) -> bool:
        return tag in self.functional_tags


class CardSemanticCatalog:
    def __init__(self, db_path=GAME_SEMANTIC_INDEX_DB):
        self._db_path = db_path

    def _ensure_ready(self) -> None:
        if self._db_path.exists():
            return
        from data.build_card_semantic_index import build_card_semantic_index
        build_card_semantic_index(db_path=self._db_path)

    def _connect(self):
        self._ensure_ready()
        return sqlite3.connect(str(self._db_path))

    @staticmethod
    def _normalize_card_id(card_id: str) -> str:
        return _slugify(str(card_id or "")).lower()

    @staticmethod
    def _vec_from_tags(tags: tuple[str, ...]) -> tuple[float, ...]:
        vec = [0.0] * len(FUNCTIONAL_TAGS)
        for tag in tags:
            idx = FUNCTIONAL_TAG_TO_IDX.get(tag)
            if idx is not None:
                vec[idx] = 1.0
        return tuple(vec)

    def mean_functional_tag_vec(self, card_ids: Iterable[str]) -> tuple[float, ...]:
        vec = [0.0] * len(FUNCTIONAL_TAGS)
        count = 0
        for card_id in card_ids:
            snap = self.get(card_id)
            if not snap.card_id:
                continue
            count += 1
            for i, v in enumerate(snap.functional_tag_vec):
                vec[i] += float(v)
        if count <= 0:
            return tuple(vec)
        return tuple(v / count for v in vec)

    @lru_cache(maxsize=4096)
    def get(self, card_id: str) -> CardSemanticSnapshot:
        cid = self._normalize_card_id(card_id)
        if not cid:
            return CardSemanticSnapshot()
        con = self._connect()
        try:
            row = con.execute(
                "SELECT id,class_name,card_type,rarity,target_type,base_cost,is_x_cost,gains_block,"
                " source_tags_json,source_keywords_json,functional_tags_json,source_path "
                "FROM cards WHERE id=? LIMIT 1",
                (cid,),
            ).fetchone()
        finally:
            con.close()
        if row is None:
            return CardSemanticSnapshot(card_id=cid)
        tags = tuple(json.loads(row[8] or "[]"))
        keywords = tuple(json.loads(row[9] or "[]"))
        functional_tags = tuple(json.loads(row[10] or "[]"))
        return CardSemanticSnapshot(
            card_id=row[0],
            class_name=row[1],
            card_type=row[2],
            rarity=row[3],
            target_type=row[4],
            base_cost=int(row[5]),
            is_x_cost=bool(row[6]),
            gains_block=bool(row[7]),
            db_tags=tags,
            db_keywords=keywords,
            functional_tags=functional_tags,
            functional_tag_vec=self._vec_from_tags(functional_tags),
            source_path=row[11],
        )


CARD_SEMANTICS = CardSemanticCatalog()
