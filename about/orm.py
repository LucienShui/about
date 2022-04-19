from peewee import Model, SqliteDatabase, AutoField, DateTimeField, TextField, CharField, DoubleField
from datetime import datetime

db = SqliteDatabase('sqlite.db')


class BaseModel(Model):
    class Meta:
        database = db


class Record(BaseModel):
    id = AutoField()

    query = TextField()
    answer = TextField()
    hit_question = TextField()
    score = DoubleField()
    knowledge_name = TextField()
    ip = CharField(max_length=39)
    created_at = DateTimeField(default=datetime.now)


db.connect()
db.create_tables([Record])
