import logging
import re
from dataclasses import dataclass

import microcore as mc
from sqlalchemy import text

from db_talk.async_db import db as async_db
from db_talk.db import db

_SQL_PATTERNS = [
    re.compile(r"```sql(.*?)```", re.IGNORECASE | re.DOTALL),
    re.compile(r"<sql>(.*?)</sql>", re.IGNORECASE | re.DOTALL),
]


def is_sql(text):
    uc_text = text.upper()
    return "SQL" in uc_text and "SELECT" in uc_text


def extract_sql(text: str) -> str:
    for pattern in _SQL_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
    return text


@dataclass(frozen=True, slots=True)
class LMSQLExecResult:
    sql: str
    columns: list[str]
    rows: list[tuple]
    error: Exception | None = None

    @property
    def no_error(self) -> bool:
        return self.error is None

    def is_empty(self) -> bool:
        return not self.rows


def execute_sql_from_response(llm_response: str):
    sql = extract_sql(llm_response)
    error, columns, rows = None, None, None
    with db().session() as ses:
        try:
            result = ses.execute(text(sql))
            columns = result.keys()
            rows = result.fetchall()
        except Exception as e:
            logging.info(mc.ui.blue(sql))
            logging.error("SQL execution error: " + str(e))
            error = e
    return LMSQLExecResult(sql=sql, columns=columns or [], rows=rows or [], error=error)


async def async_execute_sql_from_response(llm_response: str):
    sql = extract_sql(llm_response)
    error, columns, rows = None, None, None
    async with async_db().session() as ses:
        try:
            result = await ses.execute(text(sql))
            columns = result.keys()
            rows = result.fetchall()
        except Exception as e:
            logging.info(mc.ui.blue(sql))
            logging.error("SQL execution error: " + str(e))
            error = e
    return LMSQLExecResult(sql=sql, columns=columns or [], rows=rows or [], error=error)
