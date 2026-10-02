"""
보충 자료 -- 퍼셉트론 하나는 왜 XOR을 못 푸는가 (그림으로 보기)

흐름:
  1) 입력 4가지 (0,0) (0,1) (1,0) (1,1) 를 가로=입력1, 세로=입력2 인 평면 위의 "점"으로 찍는다.
  2) 뉴런의 판단식(가중합 > 문턱값)이 평면 위의 "직선 하나"가 된다는 걸 AND, OR로 확인한다.
  3) XOR은 어떤 직선을 그어도 틀리는 점이 생긴다는 걸 확인한다.
결과 그림: linear_separable_demo.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")  # [문법] 화면에 창을 안 띄우고 파일로만 저장하는 모드
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Apple SD Gothic Neo"  # macOS 한글 폰트
plt.rcParams["axes.unicode_minus"] = False

# [문법] { "이름": 값, ... } 는 딕셔너리(dict) -- 이름표(키)로 값을 찾는 표. Swift의 [String: [Int]] 와 같은 개념.
# [AI 용어] 각 리스트는 "정답표"(label) -- 입력 4가지 각각에서 뉴런이 내야 하는 출력(0 또는 1).
points = [(0, 0), (0, 1), (1, 0), (1, 1)]
problems = {
    "AND": [0, 0, 0, 1],
    "OR":  [0, 1, 1, 1],
    "XOR": [0, 1, 1, 0],
}

xs = np.linspace(-0.5, 1.5, 50)
# [문법] np.linspace(시작, 끝, 개수) 는 시작~끝 사이를 같은 간격으로 쪼갠 숫자 50개 배열을 만든다.
# 직선을 그리려면 "가로 좌표 여러 개"가 필요해서 만드는 것.

# [AI 용어] 뉴런의 판단식 "x1 + x2 > 문턱값" 의 경계는 "x1 + x2 = 문턱값" 인 점들 -- 이게 직선이다.
# [예시] 문턱값 1.5 (AND 뉴런): x1 + x2 = 1.5 를 x2 에 대해 풀면 x2 = 1.5 - x1.
#   x1=0 이면 x2=1.5, x1=1 이면 x2=0.5 -> 이 두 점을 이으면 (0,1)(1,0) 아래쪽, (1,1) 위쪽을 가르는 직선이 된다.
lines = {
    "AND": 1.5 - xs,   # AND 뉴런의 경계선 (문턱값 1.5)
    "OR":  0.5 - xs,   # OR 뉴런의 경계선 (문턱값 0.5)
}

fig, axes = plt.subplots(1, 3, figsize=(15, 6))

# [문법] zip(axes, problems.items()) -- 그래프 칸 3개와 (이름, 정답표) 3쌍을 짝지어 한꺼번에 돈다.
# "for ax, (name, answers) in ..." 의 괄호는 중첩 튜플 언패킹: ax 하나 + (name, answers) 한 쌍을 동시에 받는다.
for ax, (name, answers) in zip(axes, problems.items()):
    for (x1, x2), ans in zip(points, answers):
        if ans == 1:
            ax.scatter(x1, x2, s=500, c="tab:green", marker="o", zorder=3)
            ax.text(x1, x2, "1", ha="center", va="center", color="white", fontsize=14, zorder=4)
        else:
            ax.scatter(x1, x2, s=500, c="tab:red", marker="s", zorder=3)
            ax.text(x1, x2, "0", ha="center", va="center", color="white", fontsize=14, zorder=4)

    if name in lines:
        # AND/OR: 직선 하나로 1(초록)과 0(빨강)이 완전히 갈린다
        ax.plot(xs, lines[name], color="black", linewidth=2.5, zorder=2)
        ax.fill_between(xs, lines[name], 2, color="tab:green", alpha=0.12)  # 직선 위쪽 = "1이라고 답하는 영역"
        ax.set_title(f"{name} -- 직선 하나로 가를 수 있음", fontsize=13)
    else:
        # XOR: 그어볼 만한 직선 세 가지를 그려서 전부 틀리는 점이 생기는 걸 보여준다
        ax.plot(xs, 0.5 - xs, "--", color="gray", linewidth=2, label="OR 선: (1,1)이 틀림")
        ax.plot(xs, 1.5 - xs, "--", color="tab:blue", linewidth=2, label="AND 선: (0,1)(1,0)이 틀림")
        ax.axhline(0.5, linestyle="--", color="tab:purple", linewidth=2, label="가로선: (1,1)이 틀림")
        ax.legend(loc="lower center", fontsize=9)
        ax.set_title("XOR -- 어떤 직선을 그어도 틀리는 점이 생김", fontsize=13)

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xlabel("입력1")
    ax.set_ylabel("입력2")
    ax.set_aspect("equal")

fig.suptitle("초록 동그라미 = 출력 1이어야 함 / 빨강 네모 = 출력 0이어야 함", fontsize=13, y=0.98)
# [문법] tight_layout(rect=(왼, 아래, 오른, 위)) -- 그림 내용을 이 사각형 안(0~1 비율)에만 배치해서
# 맨 위 큰 제목(suptitle)과 각 칸 제목이 겹치거나 아래 축 이름이 잘리는 걸 막는다.
fig.tight_layout(rect=(0, 0.03, 1, 0.92))
fig.savefig("linear_separable_demo.png", dpi=150)
print("saved: linear_separable_demo.png")
