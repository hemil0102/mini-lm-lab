"""
보충 자료 -- 책 45쪽 "토폴로지 정의" / 46쪽 "첫 번째 신경세포층 구축하기" 시각화
우리가 이미 만든 layer.py의 은닉층 네트워크를 그대로 그림으로 그려서
"연결(토폴로지)"과 "시냅스 강도(가중치)"가 서로 다른 것임을 눈으로 확인한다.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(10, 6))

# 위치 정의: (x, y)
inputs = {"입력1\n(숙제)": (0, 1.0), "입력2\n(청소)": (0, 0.0)}
hidden = {"AND\nt=1.5": (1.6, 1.3), "OR\nt=0.5": (1.6, 0.65), "숙제전용\nt=0.5": (1.6, 0.0)}
output = {"출력\nt=1.5": (3.2, 0.65)}

def draw_node(name, pos, color):
    circ = Circle(pos, 0.22, facecolor=color, edgecolor="black", zorder=3)
    ax.add_patch(circ)
    ax.text(pos[0], pos[1], name, ha="center", va="center", fontsize=8, zorder=4)

def draw_edge(p1, p2, weight):
    style = "-" if weight != 0 else "--"
    alpha = 1.0 if weight != 0 else 0.3
    arrow = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=12,
                              color="gray", linestyle=style, alpha=alpha, zorder=1)
    ax.add_patch(arrow)
    mx, my = (p1[0]+p2[0])/2, (p1[1]+p2[1])/2
    ax.text(mx, my + 0.08, f"w={weight}", ha="center", fontsize=7.5, color="crimson", zorder=5)

for name, pos in inputs.items():
    draw_node(name, pos, "#cfe8ff")
for name, pos in hidden.items():
    draw_node(name, pos, "#ffe6b3")
for name, pos in output.items():
    draw_node(name, pos, "#c8f7c5")

# 연결(토폴로지) + 시냅스 강도(가중치) -- layer.py의 w1, w2 그대로
i1, i2 = list(inputs.values())
h_and, h_or, h_hw = list(hidden.values())
out = list(output.values())[0]

draw_edge(i1, h_and, 1); draw_edge(i2, h_and, 1)
draw_edge(i1, h_or, 1);  draw_edge(i2, h_or, 1)
draw_edge(i1, h_hw, 1);  draw_edge(i2, h_hw, 0)   # 숙제전용은 청소 입력과 '연결은 있지만' 가중치 0
draw_edge(h_and, out, 1); draw_edge(h_or, out, 1); draw_edge(h_hw, out, 1)

ax.set_xlim(-0.6, 4.0)
ax.set_ylim(-0.5, 1.8)
ax.axis("off")
ax.set_title("책 45~46쪽 '토폴로지' + '시냅스 강도' -- layer.py를 그대로 그린 것\n"
              "실선/화살표=연결(토폴로지)   빨간 w=시냅스 강도(가중치)   점선=연결은 있지만 가중치 0 (사실상 무시)",
              fontsize=10)
fig.tight_layout()
fig.savefig("topology_demo.png", dpi=150)
print("saved: topology_demo.png")
