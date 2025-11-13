import microcore as mc
from sqlalchemy import Engine, MetaData
from sqlalchemy.exc import CompileError
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.schema import CreateTable


def _describe(
    metadata: MetaData,
    engine: Engine | AsyncEngine,
    schema: str = None,
    short: bool = True,
) -> str:
    db_name = engine.url.database
    out = f"DB Name: {db_name}\nEngine: {engine.name}\nTables:\n"
    for table_name, table in metadata.tables.items():
        try:
            create_stmt = CreateTable(table).compile(engine)
            if short:
                create_stmt = (
                    str(create_stmt)
                    .strip()
                    .replace(f"CREATE TABLE {(schema + '.') if schema else ''}", "")
                    .replace(", \n", ",\n")
                    # .replace("\n\t", " ")
                    + "\n"
                )
            out += str(create_stmt)
        except CompileError as e:
            mc.ui.error(f"Can't describe table {table_name}:{e}")
    return out


def describe(
    engine: Engine,
    schema: str = None,
    short: bool = True,
) -> str:
    metadata = MetaData()
    metadata.reflect(bind=engine, schema=schema)
    return _describe(metadata, engine, schema, short)


async def async_describe(
    engine: AsyncEngine,
    schema: str = None,
    short: bool = True,
) -> str:
    metadata = MetaData()
    async with engine.begin() as conn:
        await conn.run_sync(metadata.reflect, schema=schema)
    return _describe(metadata, engine, schema, short)
