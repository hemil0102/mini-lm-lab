"""
보충 자료 -- «XOR은 어떤 직선을 그어도 틀리는 점이 생긴다» 를 직접 눈으로 확인하기

흐름:
  1) 직선(= 뉴런 하나의 판단식)을 5가지로 바꿔 그어보고, 각 직선이 4개 점 중 몇 개를 맞히는지 센다.
     -> 초록 영역 = 그 직선이 «1이라고 답하는 쪽». 노란 고리 = 그 직선이 틀리는 점.
  2) 왜 «아무리 해도 4개 전부는 안 되는지» 를 한 장의 그림으로 보인다:
     초록 두 점을 잇는 선과 빨강 두 점을 잇는 선이 정확히 한가운데에서 X자로 만난다.
결과 그림: xor_cross_demo.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

points = [(0, 0), (0, 1), (1, 0), (1, 1)]
answers = [0, 1, 1, 0]   # [AI 용어] XOR의 정답표(label)

# [문법] 튜플 4개짜리 리스트. 각 튜플 = (설명 글, 입력1 가중치 w1, 입력2 가중치 w2, 문턱값 t)
# [AI 용어] 직선을 바꾼다 = 뉴런의 가중치와 문턱값을 바꾼다. 뉴런은 w1*x1 + w2*x2 > t 일 때 1을 낸다.
attempts = [
    ("시도 1: OR 뉴런\n(w=1,1  문턱값 0.5)",   1,  1, 0.5),
    ("시도 2: AND 뉴런\n(w=1,1  문턱값 1.5)",   1,  1, 1.5),
    ("시도 3: 비스듬히\n(w=1,-1  문턱값 0.5)",  1, -1, 0.5),
    ("시도 4: 가로선\n(w=0,1  문턱값 0.5)",     0,  1, 0.5),
    ("시도 5: 세로선\n(w=1,0  문턱값 0.5)",     1,  0, 0.5),
]

fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
axes = axes.flatten()  # [문법] 2x3 격자를 길이 6짜리 한 줄로 펼쳐서 axes[0]~axes[5]로 접근

# 칠하기용 격자: 평면 전체를 잘게 쪼갠 좌표
# [문법] np.meshgrid(a, b) : 가로 좌표 a와 세로 좌표 b로 «모든 (가로,세로) 쌍»의 격자를 만든다.
gx, gy = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))

for ax, (title, w1, w2, t) in zip(axes[:5], attempts):
    # [예시] w1=1, w2=1, t=0.5 이고 점 (1,1) 이면: 1*1 + 1*1 = 2 > 0.5 이므로 뉴런은 1이라고 답한다.
    # 그런데 XOR 정답은 0 이므로 이 점에서는 «틀림».
    ax.contourf(gx, gy, (w1 * gx + w2 * gy > t), levels=[0.5, 1.5], colors=["tab:green"], alpha=0.15)
    # [문법] contourf(..., levels=[0.5, 1.5]) : 값이 True(=1)인 구역만 칠한다 -> «이 직선이 1이라고 답하는 쪽».
    if w2 != 0:
        xs = np.linspace(-0.5, 1.5, 50)
        ax.plot(xs, (t - w1 * xs) / w2, color="black", linewidth=2.5)   # w1*x1 + w2*x2 = t 를 x2에 대해 푼 직선
    else:
        ax.axvline(t / w1, color="black", linewidth=2.5)                # w2=0 이면 세로선 (x2로 못 풀어서 따로 처리)

    right = 0
    for (x1, x2), ans in zip(points, answers):
        said = 1 if (w1 * x1 + w2 * x2 > t) else 0
        ok = (said == ans)
        right += ok   # [문법] True는 1, False는 0으로 계산되므로 += 로 «맞힌 개수»를 센다
        ax.scatter(x1, x2, s=520, c="tab:green" if ans == 1 else "tab:red",
                   marker="o" if ans == 1 else "s", zorder=3)
        ax.text(x1, x2, str(ans), ha="center", va="center", color="white", fontsize=13, zorder=4)
        if not ok:
            ax.scatter(x1, x2, s=1100, facecolors="none", edgecolors="gold", linewidths=4, zorder=5)
    ax.set_title(f"{title}\n-> 4개 중 {right}개 맞힘", fontsize=11)
    ax.set_xlim(-0.5, 1.5); ax.set_ylim(-0.5, 1.5)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_aspect("equal")
    ax.set_xlabel("입력1"); ax.set_ylabel("입력2")

# ---- 6번째 칸: 왜 4개 전부는 불가능한가 ----
ax = axes[5]
ax.plot([0, 1], [1, 0], color="tab:green", linewidth=4, zorder=2)   # 초록 두 점을 잇는 선
ax.plot([0, 1], [0, 1], color="tab:red", linewidth=4, zorder=2)     # 빨강 두 점을 잇는 선
ax.scatter(0.5, 0.5, s=600, marker="*", c="gold", edgecolors="black", zorder=5)
for (x1, x2), ans in zip(points, answers):
    ax.scatter(x1, x2, s=520, c="tab:green" if ans == 1 else "tab:red",
               marker="o" if ans == 1 else "s", zorder=3)
    ax.text(x1, x2, str(ans), ha="center", va="center", color="white", fontsize=13, zorder=4)
ax.annotate("두 선이 정확히\n한가운데서 X자로 만남", xy=(0.5, 0.5), xytext=(0.62, 0.18),
            fontsize=11, arrowprops=dict(arrowstyle="->"))
ax.set_title("왜 4개 전부는 불가능한가\n초록끼리 잇는 선 / 빨강끼리 잇는 선이 교차함", fontsize=11)
ax.set_xlim(-0.5, 1.5); ax.set_ylim(-0.5, 1.5)
ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_aspect("equal")
ax.set_xlabel("입력1"); ax.set_ylabel("입력2")

fig.suptitle("XOR: 초록 동그라미=1이어야 함 / 빨강 네모=0이어야 함 / 노란 고리=그 직선이 틀리는 점", fontsize=13, y=0.99)
fig.tight_layout(rect=(0, 0.02, 1, 0.95))
fig.savefig("xor_cross_demo.png", dpi=140)
print("saved: xor_cross_demo.png")
