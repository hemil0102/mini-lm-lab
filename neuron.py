"""
Stage 1 -- 인공 뉴런 하나
책 44~51쪽에서 설명한 뉴런을 그대로 코드로 만든다:
- 입력(inputs) 여러 개
- 입력마다 다른 가중치(weights) = 시냅스 강도
- 가중치 합이 문턱값(threshold)을 넘으면 출력 1, 아니면 0
"""

def neuron(inputs, weights, threshold):
    # zip(inputs, weights): 두 리스트를 짝지어 (입력, 가중치) 쌍으로 함께 순회한다
    weighted_sum = sum(x * w for x, w in zip(inputs, weights))
    return 1 if weighted_sum > threshold else 0


if __name__ == "__main__":
    # AND 게이트를 흉내내는 뉴런: 둘 다 1일 때만 출력 1
    weights = [1, 1]
    threshold = 1.5

    test_cases = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for a, b in test_cases:
        result = neuron([a, b], weights, threshold)
        print(f"입력=({a},{b}) -> 가중합={a*1 + b*1} -> 출력={result}")
