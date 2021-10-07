import pandas as pd
import numpy as np
from pyecharts.charts import Bar
from pyecharts.charts import Pie
from pyecharts import options as opts
from pyecharts.globals import ThemeType
from pyecharts.globals import WarningType
from pyecharts.options import LabelOpts, LineStyleOpts

WarningType.ShowWarning = False

# zip_object = zip(x_data, y_data)
# print(zip_object.__next__())
# print(zip_object.__next__())
# print(zip_object.__next__())
# print(zip_object.__next__())
# print(zip_object.__next__())
# # StopIteration
# print(zip_object.__next__())

data = pd.read_csv(r"./practice.csv", encoding="GBK", header=None)
data.columns = ["省市", "区", "平均收入"]
data = data.iloc[:, [0, 2]]
data = data["平均收入"].groupby(data["省市"]).mean().to_frame()
data = data.reset_index()
list_data = data.values.tolist()
print(list_data)

pie1 = (
    Pie(init_opts=opts.InitOpts())
        .add(
        # 系列名称，即该饼图的名称
        series_name="地区平均收入",
        # 系列数据项，格式为[(key1,value1),(key2,value2)]
        data_pair=list_data,
        # 饼图半径 数组的第一项是内半径，第二项是外半径
        radius=["20%", "60%"],
        # 饼图的圆心，第一项是相对于容器的宽度，第二项是相对于容器的高度
        center=["50%", "50%"],
        # 标签配置项
        label_opts=opts.LabelOpts(is_show=True)
    )
        .set_colors(["blue", "cyan", "#00BFFF", "#ADD8E6"])  # LightBLue 淡蓝 #ADD8E6 rgb(173, 216, 230)
        .set_global_opts(title_opts=opts.TitleOpts(title="城市平均收入情况"))
        .set_series_opts(label_opts=opts.LabelOpts(formatter="{b} : {d}%"))
        .render(r"E:\python_data_analysis\pyecharts_html\demo1.html")
)

"""
class InitOpts(BasicOpts):
    def __init__(
        self,
        width: str = "900px",
        height: str = "500px",
        theme: str = ThemeType.WHITE,
    ):
    
ThemeType = _ThemeType()

class _ThemeType:
    BUILTIN_THEMES = ["light", "dark", "white"]
    LIGHT = "light"
    DARK = "dark"
    WHITE = "white"
    CHALK: str = "chalk"
    ESSOS: str = "essos"
    INFOGRAPHIC: str = "infographic"
    MACARONS: str = "macarons"
    PURPLE_PASSION: str = "purple-passion"
    ROMA: str = "roma"
    ROMANTIC: str = "romantic"
    SHINE: str = "shine"
    VINTAGE: str = "vintage"
    WALDEN: str = "walden"
    WESTEROS: str = "westeros"
    WONDERLAND: str = "wonderland"
    HALLOWEEN: str = "halloween"
"""

x_data = ["不及格", "60-69分", "70-79分", "80-89分", "90-100分"]
y_data_one = [1, 3, 5, 12, 8]
y_data_two = np.array([1, 2, 13, 8, 10])
# 饼图用的数据格式是[(key1, value1), (key2, value2)]，所以先使用zip函数将二者进行组合
data_pair = [list(z) for z in zip(x_data, y_data_two)]
# print(data_pair)

bar1 = (
    Bar(init_opts=opts.InitOpts(theme=ThemeType.SHINE, width="1600px", height="800px"))
        .add_xaxis(x_data)
        .add_yaxis("27班成绩分布", y_data_one, stack="stack1", color='red')
        .add_yaxis("28班成绩分布", y_data_two.tolist(), stack="stack2", color="#00BFFF")
)

bar1.set_global_opts(
    title_opts=opts.TitleOpts(title="Bar-基本示例", subtitle="ThemeType.SHINE"),
    legend_opts=opts.LegendOpts(pos_top="5%"),
    # DataZoom 添加滑块，数据区域缩放
    datazoom_opts=[opts.DataZoomOpts(is_show=True)]
)

bar1.set_series_opts(
    label_opts=opts.LabelOpts(formatter="{b} : {c}", is_show=True)
)

bar1.set_series_opts(
    markpoint_opts=opts.MarkPointOpts(
        data=[opts.MarkPointItem(type_="max", name="最大值"),
              opts.MarkPointItem(type_="min", name="最小值")]
    ),
    markline_opts=opts.MarkLineOpts(
        data=[opts.MarkLineItem(type_="average", name="平均值")],
        linestyle_opts=LineStyleOpts(color="#FF6600",
                                     # color='rgba(0, 255, 0, 0.3)',
                                     type_="dotted",  # 'solid','dashed','dotted'
                                     width=2),
        # DeepSkyBlue 深天蓝  #00BFFF  rgb(0, 191, 255)
        label_opts=LabelOpts(color="#00BFFF", font_size=20),
    )
)

bar1.render(r"E:\python_data_analysis\pyecharts_html\demo2.html")
