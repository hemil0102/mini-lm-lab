"""
보충 자료 -- «이미지판 XOR»: 가로줄 vs 세로줄 (퍼셉트론은 왜 그림을 못 맞췄나)

흐름:
  1) 3x3 흑백 그림 6장을 만든다: 세로줄 3장(정답 1), 가로줄 3장(정답 0).
     사람은 한눈에 구분하지만, 뉴런 하나(픽셀마다 가중치를 곱해 더하는 방식)로는 구분이 가능한가?
  2) 가중치를 -2~2 정수로 전부 대입(5의 9제곱 = 약 195만 가지)해서 6장을 다 맞히는 조합이 있는지 센다.
  3) 은닉층을 쌓은 손 설계 신경망(세로줄 감지 뉴런 3개 -> OR)으로는 6장을 다 맞히는지 확인한다.
"""
import itertools
import numpy as np
from layer import layer  # [문법] 같은 폴더의 layer.py 안의 layer 함수를 가져다 쓴다 (2단계에서 만든 것)

# ---- 1. 그림 6장 만들기 ----
# [문법] np.zeros((3, 3), dtype=int) : 0으로 채운 3행x3열 정수 배열 = 전부 흰 그림.
# [문법] img[:, c] = 1 : ":"는 «모든 행», c는 «c번째 열» -> 그 열 전체를 1(검정)로 칠한다.
# [예시] c=1 이면 가운데 세로줄 그림: [[0,1,0],[0,1,0],[0,1,0]].  img[r, :] = 1 은 r번째 행 전체(가로줄).
# [문법] .flatten() : 3x3 배열을 길이 9짜리 한 줄로 펼침 (뉴런은 입력을 한 줄로 받으니까).
# [AI 용어] 이렇게 2차원 그림을 1차원으로 펼치는 걸 flatten(평탄화)라고 한다. (STUDY.md의 n차원 배열 참고)
vertical, horizontal = [], []
for k in range(3):
    img = np.zeros((3, 3), dtype=int); img[:, k] = 1; vertical.append(img.flatten())
    img = np.zeros((3, 3), dtype=int); img[k, :] = 1; horizontal.append(img.flatten())
V = np.array(vertical)      # shape (3, 9) -- 세로줄 그림 3장, 정답 1
H = np.array(horizontal)    # shape (3, 9) -- 가로줄 그림 3장, 정답 0

# ---- 2. 뉴런 하나로 가능한가? 가중치를 전부 대입해서 확인 ----
# [AI 용어] 뉴런 하나의 판단 = (픽셀 9개 x 가중치 9개)의 합 > 문턱값. 6장을 다 맞히려면
#   «세로줄 3장의 합 점수가 전부 문턱값 초과» 이고 «가로줄 3장은 전부 문턱값 이하» 여야 한다.
#   => (세로줄 점수 중 가장 낮은 것) > (가로줄 점수 중 가장 높은 것) 이면 그 사이에 문턱값을 놓을 수 있다.
values = np.arange(-2, 3)  # [-2, -1, 0, 1, 2]
# [문법] itertools.product(values, repeat=9) : values에서 9번 뽑는 모든 조합(중복 허용)을 하나씩 만들어줌.
# [예시] values=[0,1], repeat=2 이면 (0,0) (0,1) (1,0) (1,1) 네 가지. 여기선 5의 9제곱 = 1,953,125가지.
weights_grid = np.array(list(itertools.product(values, repeat=9)), dtype=np.int32)  # shape (1953125, 9)
pos_scores = weights_grid @ V.T   # shape (1953125, 3) -- 가중치 조합마다 세로줄 3장의 점수
neg_scores = weights_grid @ H.T   # shape (1953125, 3) -- 가로줄 3장의 점수
# [문법] .min(axis=1) : 각 행(가중치 조합 하나)에서 열 방향으로 가장 작은 값만 뽑는다.
separable = pos_scores.min(axis=1) > neg_scores.max(axis=1)
print(f"뉴런 하나로 6장을 다 맞히는 가중치 조합: {separable.sum()}개 (검사한 조합 {len(weights_grid):,}개)")

# ---- 3. 은닉층을 쌓으면? (손 설계) ----
# 은닉 뉴런 k = «k번째 열의 픽셀 3개가 전부 검정인가?» (3개 합이 2.5 초과 = 3개 전부 1)
w1 = np.zeros((9, 3), dtype=int)
for k in range(3):
    for r in range(3):
        w1[r * 3 + k, k] = 1   # 3x3을 펼친 9칸 중 k번째 열에 해당하는 칸 3개에 가중치 1
t1 = np.array([2.5, 2.5, 2.5])
w2 = np.array([[1], [1], [1]]); t2 = np.array([0.5])  # 출력 뉴런 = OR («세로줄 감지 뉴런이 하나라도 켜졌나?»)
print("\n은닉층 신경망 (손 설계):")
for name, imgs in [("세로줄", V), ("가로줄", H)]:
    for k, x in enumerate(imgs):
        hidden = layer(x, w1, t1)
        out = layer(hidden, w2, t2)
        print(f"  {name} {k}번 -> 은닉층 {hidden} -> 출력 {out[0]}")
