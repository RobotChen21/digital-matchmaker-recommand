from datetime import datetime, date

def calc_age(birthday_val):
    if not birthday_val: return 0
    try:
        # 统一转为 date 对象进行计算
        if isinstance(birthday_val, datetime):
            b_date = birthday_val.date()
        elif isinstance(birthday_val, date):
            b_date = birthday_val
        else:
            return 0

        today = date.today()
        return today.year - b_date.year - ((today.month, today.day) < (b_date.month, b_date.day))
    except:
        return 0