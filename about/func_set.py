from datetime import datetime

from typing import Callable


def now_date() -> str:
    return datetime.now().strftime('现在时间是北京时间 %Y 年 %m 月 %d 日 %H 点 %M 分')


func_list: [Callable] = [now_date]
func_map: {str, Callable} = {func.__name__: func for func in [func_list]}
