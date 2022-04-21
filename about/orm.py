from peewee import Model, MySQLDatabase, AutoField, DateTimeField, TextField, CharField
import json
from datetime import datetime
from .tools import trace_id

db = MySQLDatabase('database', user='username', passwd='password', host='localhost')


class BaseModel(Model):
    class Meta:
        database = db


class JSONField(TextField):
    def db_value(self, value: dict) -> str:
        return json.dumps(value)

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
