"""
실험 도구 -- 손으로 설계한 XOR 신경망에 «내가 원하는 입력»을 넣어보기

흐름:
  1) 은닉층 [OR 뉴런, AND 뉴런] + 출력 뉴런(OR - AND > 0.5) 으로 된 XOR 신경망을 만든다.
     (2단계 layer.py 의 layer 함수를 그대로 가져다 쓴다)
  2) PROBES 목록의 입력을 하나씩 넣고, 은닉층 출력과 최종 출력을 찍는다.
  3) 사람이 «당연히 이 답이겠지» 하는 기대값과 비교해서, 신경망이 속는 입력을 찾는다.

사용법: 맨 아래 MY_PROBES 에 (입력1, 입력2) 를 적고 실행(VS Code 의 ▶)하면 된다.
"""
import numpy as np
from layer import layer   # [문법] 같은 폴더 layer.py 안의 layer 함수를 가져온다 (2단계에서 만든 «뉴런 한 층» 계산)

# ---- 손으로 설계한 XOR 신경망 (layer.py 에서 쓴 것과 같은 방식) ----
w1 = np.array([[1, 1], [1, 1]])    # [AI 용어] 은닉층 가중치: 입력 2개 -> 은닉 뉴런 2개 [OR, AND]
t1 = np.array([0.5, 1.5])          # [AI 용어] 은닉층 문턱값: OR 은 0.5, AND 는 1.5
w2 = np.array([[1], [-1]])         # [AI 용어] 출력 뉴런 가중치: OR 는 +1, AND 는 -1 (음수 가중치 = «억제»)
t2 = np.array([0.5])               # 출력 문턱값

def xor_net(a, b):
    hidden = layer(np.array([a, b]), w1, t1)   # [AI 용어] 순전파 1단계: 입력 -> 은닉층
    out = layer(hidden, w2, t2)                # [AI 용어] 순전파 2단계: 은닉층 -> 출력
    return hidden, int(out[0])

def human_xor(a, b):
    # 사람의 기대값: 각 입력을 «0.5 보다 크면 켜짐»으로 보고, 둘 중 정확히 하나만 켜졌을 때 1.
    # [예시] (2, 0): 2 > 0.5 는 켜짐, 0 > 0.5 는 꺼짐 -> 정확히 하나 켜짐 -> 기대값 1
    # [문법] (a > 0.5) != (b > 0.5) 는 «True/False 두 값이 서로 다른가» -> 정확히 하나만 True 일 때 True.
    return int((a > 0.5) != (b > 0.5))

PROBES = [
    (0, 0), (0, 1), (1, 0), (1, 1),   # 훈련용으로 쓴 정상 입력 4가지
    (0.4, 0.4),                        # 둘 다 «약하게» 켜짐
    (0.6, 0.6),                        # 둘 다 «애매하게» 켜짐
    (2, 0), (0, 5), (100, 0),          # 한쪽만 아주 «세게» 켜짐
    (-3, 2),                           # 음수가 섞임
]
MY_PROBES = [
    # 여기에 직접 실험하고 싶은 입력을 적어보세요. 예: (0.3, 0.3), (7, 7), (1, -1)
]

print(f"{'입력':>12} | 은닉층[OR,AND] | 신경망 | 사람 기대 | 판정")
# [문법] f"{'입력':>12}" 의 ">12" 는 «12칸 폭으로 오른쪽 정렬». 표 모양을 맞추려는 서식 지정.
for a, b in PROBES + MY_PROBES:
    hidden, out = xor_net(a, b)
    want = human_xor(a, b)
    verdict = "일치" if out == want else "<-- 속았음"   # [문법] «A if 조건 else B» 삼항 표현식
    print(f"{str((a, b)):>12} |    {hidden}    |   {out}    |    {want}     | {verdict}")
