"""
Stage 2 -- 뉴런 여러 개 = 층(layer)
1단계 뉴런 하나를, numpy로 "여러 개를 한 번에" 계산하도록 확장한다.
책 44쪽 그림의 신경망 1층/중간층/출력층이 바로 이 구조다.
"""
import numpy as np


def layer(inputs, weights, thresholds):
    """
    inputs:     shape (입력개수,)
    weights:    shape (입력개수, 뉴런개수) -- weights[:, j] 는 j번째 뉴런의 가중치
    thresholds: shape (뉴런개수,)          -- 뉴런마다 다른 기준값 가능
    반환:       shape (뉴런개수,)          -- 뉴런마다 0 또는 1
    """
    # inputs @ weights : 모든 뉴런의 "가중합"을 행렬곱 한 번으로 계산
    weighted_sums = inputs @ weights
    # 뉴런마다 자기 threshold와 비교 -- 벡터 전체에 한 번에 적용됨(브로드캐스팅)
    return (weighted_sums > thresholds).astype(int)


if __name__ == "__main__":
    # 1단계에서 따로 만들었던 AND(threshold=1.5)와 OR(threshold=0.5) 뉴런을
    # "같은 입력을 보는 한 층"으로 합쳐서 한 번에 계산해본다.
    weights = np.array([
        [1, 1],  # 입력1 -> [AND뉴런 가중치, OR뉴런 가중치]
        [1, 1],  # 입력2 -> [AND뉴런 가중치, OR뉴런 가중치]
    ])
    thresholds = np.array([1.5, 0.5])

    print("입력\t\tAND뉴런\tOR뉴런")
    for a, b in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        out = layer(np.array([a, b]), weights, thresholds)
        print(f"({a},{b})\t\t{out[0]}\t{out[1]}")

    print()
    print("-- 여러 층 쌓기: 은닉층 3개짜리 신경망 (전부 직접 정한 가중치, 손 계산 가능) --")
    # 은닉층 뉴런 3개: [AND뉴런, OR뉴런, "숙제만 보는" 뉴런]
    w1 = np.array([
        [1, 1, 1],  # 입력1(숙제) -> [AND, OR, 숙제전용]
        [1, 1, 0],  # 입력2(청소) -> [AND, OR, 숙제전용] (숙제전용은 청소를 아예 안 봄)
    ])
    t1 = np.array([1.5, 0.5, 0.5])

    # 출력층 뉴런 1개: 은닉층 3개 중 2개 이상 켜져야 최종 허락
    w2 = np.array([[1], [1], [1]])
    t2 = np.array([1.5])

    for a, b in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        inputs = np.array([a, b])
        hidden = layer(inputs, w1, t1)
        output = layer(hidden, w2, t2)
        print(f"입력=({a},{b}) -> 은닉층 출력={hidden} -> 최종 출력={output[0]}")
