from peewee import Model, MySQLDatabase, AutoField, DateTimeField, TextField, CharField, DoubleField
from datetime import datetime

db = MySQLDatabase('database', user='username', passwd='password', host='localhost')


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
