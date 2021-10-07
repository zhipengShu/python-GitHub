import matplotlib.pyplot as plt
import numpy as np

# 解决中文乱码：plt.rcParams["font.sans-serif"]=["SimHei"]
# 解决负号乱码：plt.rcParams['axes.unicode_minus'] = False

# plt.legend()使用loc参数
# 0: 'best'
# 1: 'upper right'
# 2: 'upper left'
# 3: 'lower left'
# 4: 'lower right'
# 5: 'right'
# 6: 'center left'
# 7: 'center right'
# 8: 'lower center'
# 9: 'upper center'
# 10: 'center'

# marker
# '.'       point marker
# 'o'       circle marker
# 'v'       triangle_down marker
# '^'       triangle_up marker
# '<'       triangle_left marker
# '>'       triangle_right marker
# 's'       square marker
# '*'       star marker
# '+'       plus marker
# 'x'       x marker
# 'D'       diamond marker

# line style
# '-'       solid line style
# '--'      dashed line style
# '-.'      dash-dot line style
# ':'       dotted line style

# np.random.rand(d0, d1, ..., dn)
# Create an array of the given shape and populate it with
# random samples from a uniform distribution over [0, 1).

# np.random.randn(d0, d1, ..., dn)
# Return a sample (or samples) from the "standard normal" distribution.
# d0, d1, ..., dn : int, optional
# The dimensions of the returned array, must be non-negative.
# If no argument is given a single Python float is returned.

# np.random.randint()

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False

# 定义figure画布 这是第一张画布
fig1 = plt.figure(figsize=(10, 8), num=1)
# 建立 2 × 2 相同大小的四张子图
ax1 = plt.subplot(2, 1, 1)

plt.title("成绩表单")
plt.xlabel("学生姓名")
plt.ylabel("学生成绩")

x = np.arange(12) + 1
y1 = np.random.rand(12) * 12
y2 = np.random.rand(12) * 12
list_stu = []
for i in range(len(x)):
    list_stu.append("stu" + " " + str(x[i]))
plt.xticks(x, list_stu)
# plt.yticks()传一个空元组tuple进去，可以不显示y轴刻度
plt.yticks(())

# 设置横纵坐标轴范围
plt.xlim((0, len(x) + 1))
plt.ylim((-20, 20))
plt.bar(x, +y1, facecolor="#9999ff", edgecolor="white", label="加分项成绩", alpha=1)
plt.bar(x, -y2, facecolor="#ff9999", edgecolor="white", label="扣分项成绩")

# 在指定坐标位置处添加文字
for u, v in zip(x, y1):
    # ha horizontal alignment
    plt.text(u, v + 0.5, "%.1f" % v, ha="center", va="bottom")
for u, w in zip(x, y2):
    plt.text(u, -w - 1, "-" * 1 + "%.1f" % w, ha="center", va="top")

plt.legend(loc='best')

ax2 = plt.subplot(2, 2, 3)

plt.title("成绩表单")
plt.xlabel("学生姓名")
plt.ylabel("学生成绩")

# x = np.arange(5) - 2
# x = np.linspace(-2, 2, 5)
x = [-2, -1, 0, 1, 2]
# y = [60, 75, 90, 65, 85]
y = np.array([60, 75, 90, 65, 85])
# 更改坐标轴的输出显示内容，用指定内容替换
# plt.xticks(x)
name_x = [r"$stu\ 1$", r"$stu\ 2$", r"$stu\ 3$", r"$stu\ 4$", r"$stu\ 5$"]
name_x_label = [r"$stu\ {}$".format(i + 3) for i in x]
name_y = ["bad", "pass", "normal", "good", "very\ good"]
name_y_label = [r"${}$".format(j) for j in name_y]
plt.xticks(x, name_x_label, fontsize=10, rotation=45)
plt.yticks([40, 60, 75, 90, 100], name_y_label)
# 设置横纵坐标轴范围
plt.xlim((-3, 3))
plt.ylim((20, 110))
plt.bar(x, y, color="g", label="student 1")
# 在指定坐标位置处添加文字
for i in range(len(x)):
    plt.text(i - 2, y[i] + 1, y[i])
# 添加图例
plt.legend(loc='upper left')
# 显示网格
plt.grid(True)

ax3 = plt.subplot(2, 2, 4)

# Fixing random state for reproducibility
np.random.seed(0)

data = {'a': np.arange(6),
        'b': np.arange(6),
        # randint(low, high=None, size=None, dtype=int)
        # Return random integers from `low` (inclusive) to `high` (exclusive)
        'c': np.random.randint(0, 10, size=6) * 5,
        'd': np.random.randn(6)
        }

data['d'] = np.abs(data['d']) * 100
# print(data)
plt.scatter('a', 'b', c='c', s='d', data=data)
plt.colorbar()
plt.xlabel('entry a')
plt.ylabel('entry b')
plt.title("data关键字参数:传入一个字典")
# 设置刻度
plt.axis([-1, 6, -1, 6])

# 定义子图间距，分布美观
# plt.tight_layout(pad=3)
plt.subplots_adjust(top=0.9, bottom=0.1, left=0.12, right=0.9, hspace=0.3, wspace=0.3)

# 保存图片
# plt.savefig(r"E:\python_data_analysis\matplotlib_image\image{}.png".format(1))

# # 定义figure画布 这是第二张画布
# fig2 = plt.figure(figsize=(8, 6), num=2)
#
# axs = fig2.subplots(2, 2)
# # ax1 = axs[0][0] 等价于 ax1 = axs[0, 0]
# ax1 = axs[0][0]
# ax2 = axs[0][1]
# ax3 = axs[1, 0]
# ax4 = axs[1, 1]
#
# x2 = list(range(10, 100, 10))
# y2 = [i ** 2 for i in x2]
#
# ax1.scatter(np.arange(4), np.arange(4))
# ax1.set_title("haha")
# ax2.plot(x2, y2, linewidth='1', label="test2", color='b', linestyle='--', marker='.')
# ax2.legend(loc='upper left')
# ax3.scatter(np.arange(4), np.arange(4))
# # plt 默认在最后一张子图上绘制，即默认在ax4上作图
# plt.scatter(np.arange(4) + 1, np.arange(4) + 1)
# # fontdict={"size": 16, "color": "r"} 字典可修改字体大小，字体颜色等等
# plt.text(1.2, 3.7, r"$\mu_n,\alpha_t\ and\ \sigma_j$", fontdict={"size": 16, "color": "r"})

# # 定义figure画布 这是第三张画布
# fig3 = plt.figure(figsize=(8, 6), num=3)
#
# ax1 = plt.subplot(2, 1, 1)
# plt.plot(np.arange(5) - 2, np.arange(5) - 2)
# axis1 = plt.gca()
# axis1.spines["right"].set_color("none")
# axis1.spines["top"].set_color("none")
# axis1.xaxis.set_ticks_position("bottom")
# axis1.yaxis.set_ticks_position("left")
# axis1.spines["bottom"].set_position(("data", 0))
# axis1.spines["left"].set_position(("data", 0))
#
# ax2 = plt.subplot(2, 2, 3)
# x1 = np.random.rand(12).reshape((3, 4))
# print(x1)
# plt.imshow(x1, interpolation="nearest", origin="upper")
# plt.colorbar(shrink=0.9)
# 
# ax3 = plt.subplot(2, 2, 4)

# # 定义figure画布 这是第四张画布
# fig4 = plt.figure(figsize=(10, 8), num=4)
#
# ax1 = plt.subplot2grid((4, 4), (0, 0), rowspan=2, colspan=2)
# ax1.set_title("ax1 demo")
# ax2 = plt.subplot2grid((4, 4), (0, 2), rowspan=2, colspan=1)
# ax3 = plt.subplot2grid((4, 4), (0, 3), rowspan=2, colspan=1)
# ax4 = plt.subplot2grid((4, 4), (2, 0), rowspan=2, colspan=4)
#
# plt.tight_layout(pad=2)

# # 定义figure画布 这是第五张画布
# fig5 = plt.figure(figsize=(10, 8), num=5)
#
# x3 = [1, 2, 3, 4, 5, 6]
# # [1, 4, 9, 16, 25, 36]
# y3 = [i ** 2 for i in x3]
# y4 = y3[::-1]  # [36, 25, 16, 9, 4, 1]
#
# left, bottom, width, height = 0.05, 0.05, 0.9, 0.9
# plt.axes([left, bottom, width, height])
# plt.plot(x3, y3, color="g", linestyle="-.", marker="D")
#
# plt.axes([0.1, 0.5, 0.4, 0.4])
# plt.title("subgraph 1")
# plt.plot(y3, x3, color="b", linestyle="--", marker="*")
#
# plt.axes([0.6, 0.1, 0.3, 0.3])
# plt.title("subgraph 2")
# plt.plot(x3, y4, color="b", linestyle="-", marker="*")

plt.show()
