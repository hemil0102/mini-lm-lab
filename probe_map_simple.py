"""
쉬운 버전 -- «신경망에게 던질 수 있는 모든 질문의 지도» 위에 대표 질문 몇 개만 찍어서 보기

읽는 법:
  - 지도의 점 하나 = 신경망에게 던지는 질문 하나 (가로=입력1 a, 세로=입력2 b)
  - 초록색 = 신경망이 «1» 이라고 답하는 곳, 흰색 = «0» 이라고 답하는 곳
  - 파란 점 = 정상 질문 4개 (0과 1만 쓰는 XOR 문제), 빨간 별 = 신경망이 속는 질문의 예
흐름: 1) 지도 전체를 신경망 답으로 색칠 2) 정상 질문과 문제 질문을 찍고, 신경망의 답과 사람의 기대를 글로 붙인다.
결과 그림: probe_map_simple.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from layer import layer
plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

w1 = np.array([[1, 1], [1, 1]]); t1 = np.array([0.5, 1.5])
w2 = np.array([[1], [-1]]);      t2 = np.array([0.5])

def xor_net(a, b):
    return int(layer(layer(np.array([a, b]), w1, t1), w2, t2)[0])

def human_xor(a, b):
    return int((a > 0.5) != (b > 0.5))

# 지도 색칠: 격자의 모든 점에서 신경망 답을 구한다
# [문법] 리스트 컴프리헨션 [식 for 변수 in 목록] : for 문으로 하나씩 돌며 식의 결과를 모아 리스트로 만드는 한 줄 문법.
# [예시] [x * 2 for x in [1, 2, 3]] -> [2, 4, 6].  여기선 b 한 줄(가로 121칸)을 만들고, 그 줄을 b 값마다 쌓아 2차원 지도를 만든다.
grid = np.linspace(-3, 3, 121)
net_map = np.array([[xor_net(a, b) for a in grid] for b in grid])

fig, ax = plt.subplots(figsize=(9.5, 8.5))
ax.imshow(net_map, origin="lower", extent=[-3, 3, -3, 3], cmap="Greens", vmin=0, vmax=1.4)

normal = [(0, 0), (0, 1), (1, 0), (1, 1)]
tricky = [(2, 0), (-3, 2), (0.4, 0.4)]
# [문법] 튜플 두 개를 쓰는 for 문: for (a, b), color in ... 처럼 괄호 안 값과 바깥 값을 한꺼번에 받을 수 있다.
for (a, b), color, marker, size in [(p, "blue", "o", 130) for p in normal] + [(p, "red", "*", 420) for p in tricky]:
    ax.scatter(a, b, s=size, c=color, marker=marker, zorder=5, edgecolors="white")
    net, human = xor_net(a, b), human_xor(a, b)
    verdict = "맞음" if net == human else "속음!"
    # [예시] (2, 0): 신경망 0, 사람 기대 1 -> «속음!»
    # [문법] dict(딕셔너리) {키: 값}: 점마다 «글상자를 놓을 위치»를 따로 적어 두고 키로 꺼내 쓴다. (Swift의 [Key: Value] 와 같음)
    # 글상자끼리 겹치지 않게 위치를 손으로 정했고, 화살표로 어느 점의 글인지 이어 준다.
    label_pos = {
        (0, 0): (-2.9, -1.3), (1, 0): (1.6, -2.4), (0, 1): (-2.9, 0.9), (1, 1): (1.6, 1.4),
        (2, 0): (1.7, -1.2), (-3, 2): (-2.9, 2.4), (0.4, 0.4): (-0.6, 2.3),
    }
    ax.annotate(f"({a},{b})\n신경망 {net} / 사람 {human}\n{verdict}", xy=(a, b), xytext=label_pos[(a, b)],
                fontsize=10, color="red" if verdict != "맞음" else "navy", zorder=6,
                arrowprops=dict(arrowstyle="-", color="gray"),
                bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.95))

ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
ax.set_xlabel("입력1 (a)"); ax.set_ylabel("입력2 (b)")
ax.set_title("신경망에게 던질 수 있는 «모든 질문»의 지도\n초록=신경망이 1이라고 답하는 곳 / 흰색=0이라고 답하는 곳", fontsize=12)
fig.text(0.5, 0.02, "파란 점 = 정상 질문 4개 (모두 맞음)   빨간 별 = 신경망이 속는 질문의 예 (지도엔 이런 곳이 훨씬 많음)", ha="center", fontsize=11)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig("probe_map_simple.png", dpi=140)
print("saved: probe_map_simple.png")
