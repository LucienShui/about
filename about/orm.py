import json
import os
from datetime import datetime

from peewee import Model, MySQLDatabase, SqliteDatabase, AutoField, DateTimeField, TextField, CharField

from .tools import trace_id

BASE_DIR: str = 'resource'

with open(os.path.join(BASE_DIR, 'config.json')) as file:
    config: dict = json.load(file)
    database_config: dict = config['database']
    if database_config.get('type', 'mysql') == 'mysql':
        db = MySQLDatabase(
            database_config['mysql-database'], user=database_config['mysql-username'],
            passwd=database_config['mysql-password'], host=database_config['mysql-host']
        )
    else:
        db = SqliteDatabase(os.path.join(BASE_DIR, database_config['sqlite-file']))


class BaseModel(Model):
    class Meta:
        database = db


class JSONField(TextField):
    def db_value(self, value: dict) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(',', ':'))

    def python_value(self, value: str) -> dict:
        return json.loads(value)


class Trace(BaseModel):
    id = AutoField()

    module = CharField(max_length=32, null=False, index=True)
    input = JSONField(default={})
    output = JSONField(default={})
    log = JSONField(default={})

    ip = CharField(max_length=39, null=False)
    trace_id = CharField(max_length=32, default=trace_id)
    created_at = DateTimeField(default=datetime.now)


db.connect()
db.create_tables([Trace])
