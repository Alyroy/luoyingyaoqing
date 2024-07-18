import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import re
from lunarcalendar import Converter, Solar

def split_query_prompt(row):
    # 尝试按"\n"拆分query列
    parts = row['user-query'].split('\n', 1)
    row['user-query'] = parts[0]  # 更新query列为第一部分
    row['user_prompt'] = parts[1] if len(parts) > 1 and parts[1] != '' else None  # 如果有第二部分且不为空，则更新prompt列
    return row


def random_date_between(start_date, end_date):
    """
    生成start_date和end_date之间的随机日期。
    "2024-02-01", "2024-03-31"
    """
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    # 计算两个日期之间的时间差
    delta = end_date - start_date
    # 生成一个介于0到delta.days之间的随机整数
    random_days = random.randint(0, delta.days)
    # 在开始日期上加上随机天数
    random_date = start_date + timedelta(days=random_days)
    return random_date


def process_row_and_generate_prompt(row, start_date, end_date):
    """
    对DataFrame的每一行，尝试提取最大日期，如果没有找到发布时间，则生成一个随机日期。
    然后，将这个日期（无论是找到的最大日期还是生成的随机日期）用于生成并拼接到'user-query'列。
    """
    # 尝试从observation列提取日期
    dates = re.findall(r'发布时间：(\d{4}-\d{2}-\d{2})', row['observation'])
    dates = [pd.to_datetime(date) for date in dates]
    
    if dates:
        # 如果找到了日期，取最大值
        max_date = max(dates)
        row['max_date'] = max_date
        # 生成一个0到60之间的随机整数
        random_days = random.randint(1, 30)
    else:
        # 如果没有找到日期，生成一个随机日期
        max_date = random_date_between(start_date, end_date)
        row['max_date'] = ''
        random_days = random.randint(1, 2)
    
    # 在最大日期上加上随机天数
    random_date = max_date + timedelta(days=random_days)
    # 将日期转换为指定格式的字符串
    random_date_str = random_date.strftime("%Y年%m月%d日")
    
    # 检查'user-query'是否已经包含了一些prompt信息
    if pd.isnull(row.get('user_prompt')):
        # 如果'user_prompt'为空，则使用生成的日期字符串
        row['user_prompt'] = random_date_str
    
    return row


def convert_date_format(date_str):
    # 使用正则表达式匹配日期
    match = re.search(r'(\d{4}年\d{1,2}月\d{1,2}日)', date_str)
    if match:
        # 提取日期字符串
        date_part = match.group(1)
        # 解析日期
        date_obj = datetime.strptime(date_part, '%Y年%m月%d日')
        return date_obj
    else:
        # 如果没有匹配到日期，返回原始字符串或错误信息
        print(date_str)
        raise "日期提取错误"
        

def get_lunar_date(date_obj):
    """
    date_obj: 阳历日期，datetime.datetime(2024, 2, 27, 0, 0)
    """
    # 输入的阳历日期
    solar = Solar(date_obj.year, date_obj.month, date_obj.day)
    lunar = Converter.Solar2Lunar(solar)

    # 获取农历的年、月、日
    lunar_year = lunar.year
    lunar_month = lunar.month
    lunar_day = lunar.day

    # 映射数字月份到农历月份名称
    def get_lunar_month_name(month):
        names = ['正月', '二月', '三月', '四月', '五月', '六月', '七月', '八月', '九月', '十月', '十一月', '腊月']
        return names[month - 1]

    # 转换日期到农历表示（例如，18转换为"十八"，8转换为"初八"）
    def get_lunar_day_name(day):
        if day == 10:
            return "初十"
        elif day < 10:
            return "初" + ["一", "二", "三", "四", "五", "六", "七", "八", "九"][day - 1]
        else:
            tens = ["十", "廿", "卅"][day // 10 - 1]
            ones = day % 10
            if ones == 0:
                return tens
            else:
                return tens + ["一", "二", "三", "四", "五", "六", "七", "八", "九"][ones - 1]

    # 获取农历的生肖
    def get_zodiac_sign(year):
        zodiacs = ['鼠', '牛', '虎', '兔', '龙', '蛇', '马', '羊', '猴', '鸡', '狗', '猪']
        return zodiacs[(year - 4) % 12]

    # 获取天干地支
    def get_tgdz_sign(year):
        t=(year-3)%10
        d=(year-3)%12
        tg='癸甲乙丙丁戊己庚辛壬'
        dz='亥子丑寅卯辰已午未申酉戌'
        return f'{tg[t]}{dz[d]}'

    zodiac_sign = get_zodiac_sign(lunar_year)
    tgdz_sign = get_tgdz_sign(lunar_year)

    # 拼接农历日期字符串
    lunar_month_name = get_lunar_month_name(lunar_month)
    lunar_day_name = get_lunar_day_name(lunar_day)
    lunar_date_str = f"用户今天农历：{tgdz_sign}年，{zodiac_sign}年，{lunar_month_name}{lunar_day_name}。"

    # 输出结果
    return lunar_date_str


def generate_sys_prompt(row):
    cities = ['北京', '上海', '广州', '深圳', '成都', '杭州', '重庆', '武汉', '西安', '苏州']
    
    # 获取用户提问日期
    if 'user_prompt' in row:
        date_str = row['user_prompt']
        date_obj = convert_date_format(date_str)
    # 随机生成
    else:
        start_date = "2020-01-01"
        end_date = "2026-12-31"
        date_obj = random_date_between(start_date, end_date)
    
    date_str = date_obj.strftime('用户今天日期：%Y年%m月%d日。')
    
    # 获取农历
    lunar_date_str = get_lunar_date(date_obj)
    
    # 生成随机时间
    random_hour = random.randint(0, 23)
    random_minute = random.randint(0, 59)
    time_str = f"用户现在时间：{random_hour:02d}时{random_minute:02d}分。"
    
    # 随机选择一个城市
    city = random.choice(cities)
    location_str = f"用户现在位置：中国{city}。"
    
    # 拼接最终的prompt
    final_prompt = f"你是一个名字叫做理想同学的AI数字生命体。\n理想同学是一个可靠的智能家庭助手，由理想汽车智能空间部门创造。\n理想同学能够理解人类的指令和意图，并且给出合理的、切合问题的、没有歧视、中立的、安全的回复。\n{date_str}{lunar_date_str}{time_str}{location_str}\n请根据以下文本写一个合适的回复。"
    
    # 更新row
    row['system'] = final_prompt
    
    return row


def append_fourth_line(row):
    system_lines = row['system'].split('\n')
    # Check if there are at least 4 lines
    if len(system_lines) >= 4:
        # Return the modified 'user-query' with the fourth line of 'system' appended
        return row['user-query'] + '\n' + system_lines[3]
    else:
        # Return the original 'user-query' if there are not enough lines in 'system'
        return row['user-query']

# Apply the function to each row
# df['user-query'] = df.apply(append_fourth_line, axis=1)


# file_name = 'obs_噪声摘要不相关个数4.csv'
# df = pd.read_csv(file_name)
# # split 已有的user and prompt
# # df = df.apply(split_query_prompt, axis=1)

# start_date = '2024-01-01'
# end_date = '2024-03-31'
# # 使用 lambda 函数来传递额外的参数
# df = df.apply(lambda row: process_row_and_generate_prompt(row, start_date, end_date), axis=1)
# df = df.apply(generate_sys_prompt,axis=1)
# df.to_csv(file_name,index=False)