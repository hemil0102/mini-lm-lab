"""
실험 -- «내 신경망의 약점 지도» 그리기 (블랙박스 스캔 vs 화이트박스 분석)

대상: try_input.py 의 손 설계 XOR 신경망 (은닉층 [OR, AND] -> 출력 OR - AND).
흐름:
  1) [블랙박스] 내부를 모른다고 치고, 입력 (a, b) 를 -3 ~ 3 범위에서 촘촘히 넣어 «출력 지도» 를 만든다.
  2) 사람이 기대하는 답(human_xor)과 비교해서 «신경망이 속는 구역» 지도를 만든다.
  3) [화이트박스] 가중치를 직접 읽어 «왜 그런 모양이 나오는지» 를 계산으로 설명한다.
결과 그림: probe_map_demo.png
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
    return int((a > 0.5) != (b > 0.5))   # 각 입력을 «0.5 초과면 켜짐» 으로 보고 정확히 하나만 켜졌을 때 1

# ---- 1~2. 블랙박스 스캔: 격자 전체에 입력을 넣어 출력 지도를 만든다 ----
# [문법] np.linspace(-3, 3, 121) : -3 ~ 3 을 같은 간격으로 쪼갠 121개 숫자 (간격 0.05).
# [예시] 격자 점 (a, b) = (2, 0) 이면 xor_net(2, 0) = 0, human_xor(2, 0) = 1 -> «불일치» 칸.
grid = np.linspace(-3, 3, 121)
net_map = np.zeros((len(grid), len(grid)), dtype=int)   # 신경망 출력 지도
bad_map = np.zeros((len(grid), len(grid)), dtype=int)   # 신경망이 사람 기대와 다른 칸 = 1
# [문법] enumerate(grid) : (번호, 값) 쌍을 하나씩 돌려준다. 번호는 지도의 칸 위치로, 값은 입력으로 쓴다.
for i, b in enumerate(grid):         # i = 세로(입력2) 번호
    for j, a in enumerate(grid):     # j = 가로(입력1) 번호
        net_map[i, j] = xor_net(a, b)
        bad_map[i, j] = int(net_map[i, j] != human_xor(a, b))
print(f"스캔한 입력 수: {bad_map.size:,}개,  신경망이 속는 입력: {bad_map.sum():,}개 ({bad_map.mean():.0%})")

# ---- 3. 화이트박스: 가중치를 읽어 계산으로 설명 ----
# OR 뉴런  = (a + b > 0.5), AND 뉴런 = (a + b > 1.5)  (가중치가 전부 1 이라 입력은 «합 a+b» 로만 쓰임)
# 출력 = OR - AND > 0.5  <=>  0.5 < a + b <= 1.5 일 때만 1.
# [AI 용어] 화이트박스 분석 = 모델 내부(가중치·문턱값)를 직접 보고 약점을 계산으로 찾는 것.
for s in [0.0, 0.5, 0.6, 1.5, 1.6]:
    print(f"  a+b = {s:>4} 인 입력 (예: ({s}, 0)) -> 출력 {xor_net(s, 0)}")
print("  (1000, -999.2) -> 출력", xor_net(1000, -999.2), " <- 합이 0.8 (0.5 초과 1.5 이하) 이라 1 이 나옴: 가중치를 알면 이런 엉뚱한 입력도 설계 가능")

fig, axes = plt.subplots(1, 2, figsize=(14, 6.4))
ext = [-3, 3, -3, 3]
ax = axes[0]
ax.imshow(net_map, origin="lower", extent=ext, cmap="Greens", vmin=0, vmax=1.4)
ax.set_title("① 블랙박스 스캔: 신경망이 «1»이라고 답하는 구역(초록)\n내부를 몰라도 입력을 촘촘히 넣으면 «비스듬한 띠» 모양이 드러남", fontsize=11)
ax = axes[1]
ax.imshow(bad_map, origin="lower", extent=ext, cmap="Reds", vmin=0, vmax=1.4)
ax.set_title("② 사람 기대와 다르게 답하는 구역(빨강) = 약점 지도\n정상 입력 4개(●)만 보면 완벽해 보이지만 대부분이 약점", fontsize=11)
xs = np.linspace(-3, 3, 50)
for ax in axes:
    ax.plot(xs, 0.5 - xs, "k--", linewidth=1.2)   # a + b = 0.5  (OR 문턱)
    ax.plot(xs, 1.5 - xs, "k--", linewidth=1.2)   # a + b = 1.5  (AND 문턱)
    for (a, b) in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        ax.scatter(a, b, s=90, c="blue", zorder=5)
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_xlabel("입력1 (a)"); ax.set_ylabel("입력2 (b)"); ax.set_aspect("equal")
fig.suptitle("점선 = 화이트박스로 읽어낸 경계 (a+b=0.5, a+b=1.5) / 파란 점 = 훈련에 쓴 정상 입력 4개", fontsize=12, y=0.99)
fig.tight_layout(rect=(0, 0.02, 1, 0.94))
fig.savefig("probe_map_demo.png", dpi=140)
print("saved: probe_map_demo.png")
