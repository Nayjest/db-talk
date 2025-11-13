import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import IntEnum
from textwrap import indent
from time import time

import microcore as mc
from microcore import ui
from microcore.ai_func import AiFuncSyntax, ai_func
from microcore.configuration import TRUE_VALUES
from sqlalchemy.ext.asyncio import AsyncEngine

from db_talk.async_db import db
from db_talk.execution import (
    LMSQLExecResult,
    async_execute_sql_from_response,
)
from db_talk.metadata import async_describe

_last_agent_id = 0


def new_agent_id():
    global _last_agent_id
    _last_agent_id += 1
    return str(_last_agent_id)


@ai_func(name="query")
async def query_db_with_finish(
    sql: str,
    finish_if_non_empty: bool = False,  # If True and results are non-empty, will finish the task
):
    """
    Query the database.
    - Always put limits on the number of returned rows (e.g., LIMIT 10).
    - Never fetch more than 100 rows if not explicitly requested.
    - Avoid heavy operations (e.g., JOINs on large tables) unless necessary.
      Better do multiple simple queries.
    """
    if isinstance(finish_if_non_empty, str):
        finish_if_non_empty = finish_if_non_empty.upper() in TRUE_VALUES
    db_result = await async_execute_sql_from_response(sql)
    return db_result, finish_if_non_empty


@ai_func
async def think(planning: str):
    """
    Use this function to plan your next steps before querying the database.
    - Reflect on the user's request and the current context.
    - Outline a clear plan of action.
    - Do not interact with the database in this step.
    """
    return f"Planning: {planning}"


@ai_func(name="query")
async def query_db(
    sql: str,
):
    """
    Query the database.
    - Always put limits on the number of returned rows (e.g., LIMIT 10).
    - Never fetch more than 100 rows if not explicitly requested.
    - Avoid heavy operations (e.g., JOINs on large tables) unless necessary.
      Better do multiple simple queries.
    """
    db_result = await async_execute_sql_from_response(sql)
    return db_result


@dataclass
class Profiling:
    agent_id: str = field(default="")
    total_time: float = field(default=0)
    llm_requests: int = field(default=0)
    sql_requests: int = field(default=0)
    llm_time: float = field(default=0)
    db_time: float = field(default=0)
    turns: int = field(default=0)


class Profiler:
    def __init__(self, agent_id: str | None = None):
        self.agent_id = agent_id
        self._start_time = time()
        self._llm_requests = 0
        self._sql_requests = 0
        self._llm_time = 0.0
        self._db_time = 0.0

    @contextmanager
    def track_llm(self):
        """Context manager to track LLM request time"""
        start = time()
        try:
            yield
        finally:
            self._llm_time += time() - start
            self._llm_requests += 1

    @contextmanager
    def track_sql(self):
        """Context manager to track SQL request time"""
        start = time()
        try:
            yield
        finally:
            self._db_time += time() - start
            self._sql_requests += 1

    def export(self, turns: int = 0) -> Profiling:
        return Profiling(
            total_time=time() - self._start_time,
            llm_requests=self._llm_requests,
            sql_requests=self._sql_requests,
            llm_time=self._llm_time,
            db_time=self._db_time,
            turns=turns,
            agent_id=self.agent_id,
        )


@dataclass
class AgentResponse:
    profile: Profiling | None = field()


@dataclass
class DataTable(AgentResponse):
    columns: list[str] = field()
    rows: list[tuple] = field()
    sql: str = field(default="")


class Verbosity(IntEnum):
    SILENT = 0
    MINIMAL = 1
    NORMAL = 2
    VERBOSE = 3


@dataclass
class Agent:
    engine: AsyncEngine = field(default_factory=lambda: db().engine)
    schema: str = "public"
    db_descr: str | None = field(default=None, init=False)
    max_conversation_turns: int = 10
    verbosity: Verbosity = field(default=Verbosity.NORMAL)
    id: str = field(default_factory=new_agent_id)

    use_early_finish: bool = field(default=False)
    allow_think: bool = field(default=True)

    llm: callable = field(default=None)

    async def init(self):
        if not self.db_descr:
            self.db_descr = await async_describe(db().engine, schema="public")
        if not self.llm:
            self.llm = mc.allm

    async def run(self, task: str):
        profiler = Profiler(agent_id=self.id)
        tools = mc.ai_func.ToolSet(
            [query_db_with_finish if self.use_early_finish else query_db]
        )
        drop_messages = None
        if self.allow_think:
            tools.append(think)
        system_message = mc.prompt(
            """
            # Task
            Satisfy the user's request with a comprehensive and professional response,
            iteratively using the tools provided to you as needed to ensure the best result.

            # User Request
            {{ task }}

            # Tools
            {{ tools }}

            # Database Schema
            {{ db_descr }}

            # Requirements
            - (!) Your response MUST be a SINGLE TOOL CALL {{ syntax }}
              with appropriate inner content according to the available schemas above.
              You communicate not directly with the user,
              but with middleware executing your tool calls.
            {% if thinking -%}
            - Break complex tasks into sequential steps;
              Do one atomic action ({{ syntax }}) per response.
            {%- endif %}
            """,
            task=task,
            tools=tools,
            db_descr=self.db_descr,
            syntax=mc.config().DEFAULT_AI_FUNCTION_SYNTAX,
            thinking=self.allow_think,
        ).as_system
        conv = [system_message]
        turn = 0
        last_res: LMSQLExecResult = None
        while True:
            turn += 1
            if turn > self.max_conversation_turns:
                logging.error("Maximum conversation turns reached.")
                return
            if self.verbosity >= Verbosity.NORMAL:
                logging.info(ui.magenta(f"--- Agent Turn {turn} ---"))
            with profiler.track_llm():
                answer = await self.llm(conv)
            if drop_messages:
                # Drop messages by list of indexes
                for idx in sorted(drop_messages, reverse=True):
                    logging.info(ui.yellow(f"Dropping message: {idx}:{conv[idx]}"))
                    conv.pop(idx)
                drop_messages = None

            conv.append(answer.as_assistant)
            if (
                self.allow_think
                and mc.env().config.DEFAULT_AI_FUNCTION_SYNTAX == AiFuncSyntax.TAG
            ):
                import re

                # Remove think tags from the assistant message for logging
                think_tag_pattern = re.compile(
                    r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL
                )
                answer = think_tag_pattern.sub("", answer).strip()
            if tool_params := tools.extract_tool_params(answer):

                if tool_params[0] == "think":
                    conv.append(mc.UserMsg("Continue"))
                    continue
                with profiler.track_sql():
                    t = await tools.call(*tool_params)

                if self.use_early_finish:
                    last_res = t[0]
                    finish_if_non_empty: bool = t[1]
                else:
                    last_res = t
                    finish_if_non_empty = False

                if last_res.error:
                    conv.append(
                        mc.UserMsg(
                            f"The SQL you provided is invalid.\n"
                            f"Please correct it.\n"
                            f"Error: {last_res.error}\n"
                            f"Retry with corrected SQL."
                        )
                    )
                    continue
                if self.verbosity >= Verbosity.NORMAL:
                    logging.info(mc.ui.green("SQL executed successfully."))

                if last_res.rows:
                    rows = (
                        "\n[\n"
                        + indent(",\n".join(str(row) for row in last_res.rows), "  ")
                        + "\n]"
                    )
                else:
                    rows = " " + str(list(last_res.rows))
                result_str = f"Columns: {list(last_res.columns)}\nRows:{rows}"
                tool_result_message = mc.prompt(
                    """
                    SQL Execution Results:
                    {{ result_str }}
                    Further instructions:
                    If this data is sufficient to provide a high-quality answer to the user request,
                    answer with word END in angle brackets: <END>.
                    If it is only one among multiple requested tables,
                    answer with word SEND_PARTIAL in angle brackets: <SEND_PARTIAL>.
                    Otherwise, continue to use database querying tool.
                    """,
                    result_str=result_str,
                )
                conv.append(tool_result_message.as_user)
                if self.verbosity >= Verbosity.NORMAL:
                    logging.info(ui.yellow(f"SQL Execution Results:\n{result_str}"))
                if finish_if_non_empty:
                    if self.verbosity >= Verbosity.NORMAL:
                        logging.info(
                            "Finishing task as non-empty result obtained, early exit."
                        )
                    yield DataTable(
                        columns=last_res.columns,
                        rows=last_res.rows,
                        sql=last_res.sql,
                        profile=profiler.export(turns=turn),
                    )
                    return
            else:
                if "<END>" in answer:
                    if self.verbosity >= Verbosity.NORMAL:
                        logging.info(
                            "Finishing task as non-empty result obtained, early exit."
                        )
                        logging.info("<END> tag found in LLM response. Finishing task.")
                    yield DataTable(
                        columns=last_res.columns,
                        rows=last_res.rows,
                        sql=last_res.sql,
                        profile=profiler.export(turns=turn),
                    )
                    return
                elif "<SEND_PARTIAL>" in answer:
                    if self.verbosity >= Verbosity.NORMAL:
                        logging.info(ui.magenta("Partial result sent to user."))
                    yield DataTable(
                        columns=last_res.columns,
                        rows=last_res.rows,
                        sql=last_res.sql,
                        profile=profiler.export(turns=turn),
                    )
                    conv.append(mc.UserMsg("Continue"))
                else:
                    if self.verbosity >= Verbosity.MINIMAL:
                        logging.error("No tool call found in LLM response.")
                    conv.append(
                        mc.UserMsg(
                            "Error: No tool call found. Please respond with a valid tool call."
                        )
                    )
                    drop_messages = [len(conv) - 1, len(conv) - 2]

        if self.verbosity >= Verbosity.MINIMAL:
            logging.info("Agent finished execution.")
