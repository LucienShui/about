from datetime import datetime


def now_date() -> str:
    return datetime.now().strftime('现在时间是北京时间 %Y 年 %m 月 %d 日 %H 点 %M 分')
