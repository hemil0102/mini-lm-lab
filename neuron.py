"""
Stage 1 -- 인공 뉴런 하나
책 44~51쪽에서 설명한 뉴런을 그대로 코드로 만든다:
- 입력(inputs) 여러 개
- 입력마다 다른 가중치(weights) = 시냅스 강도
- 가중치 합이 문턱값(threshold)을 넘으면 출력 1, 아니면 0

[문법] 파일 맨 위 세 겹 따옴표(\"\"\"...\"\"\")는 "모듈 docstring" -- 이 파일이 뭘 하는 파일인지
적어두는 공식 주석 자리. 그냥 # 주석과 달리 다른 코드에서 이 파일을 가져다 쓸 때
"이 파일 설명 좀 보여줘" 하면 이 부분이 나옴. Swift에는 이런 파일 맨 위 전용 자리가 따로 없음.
"""

def neuron(inputs, weights, threshold):
    # [AI 용어] 이 함수 전체가 "인공 뉴런(artificial neuron)" 하나 = 뇌의 신경세포 하나를 흉내낸 최소 계산 단위.
    # inputs=입력, weights=가중치(=시냅스 강도, 입력마다 다르게 매기는 중요도), threshold=문턱값(발화 기준).

    # zip(inputs, weights): 두 리스트를 짝지어 (입력, 가중치) 쌍으로 함께 순회한다.
    # [문법] "sum(... for ... in ...)"는 제너레이터 표현식(generator expression) --
    # 리스트를 통째로 안 만들고 값을 하나씩 계산해서 바로 sum()에 넘겨줌.
    # Swift로 치면 zip(inputs, weights).map { $0 * $1 }.reduce(0, +) 한 줄을 이렇게 쓴 것과 같음.
    #
    # [예시] inputs=[1, 0], weights=[1, 1] 이 들어오면:
    #   1) zip(inputs, weights) -> (1, 1) 과 (0, 1), 이렇게 두 쌍을 순서대로 만듦
    #   2) "x * w for x, w in ..." -> 각 쌍마다 x*w 계산: 1*1=1, 0*1=0 -> 값 두 개(1, 0)가 하나씩 만들어짐
    #   3) sum(...) -> 그 값들을 다 더함: 1 + 0 = 1
    #   즉 weighted_sum = 1 이 된다. (입력이 [1, 1]이면 1*1 + 1*1 = 2, [0, 0]이면 0)
    weighted_sum = sum(x * w for x, w in zip(inputs, weights))
    # [AI 용어] weighted_sum = "가중합" -- 각 입력에 가중치를 곱해서 다 더한 값. 뉴런이 최종 판단 전에 보는 "점수".

    # [문법] "A if 조건 else B"는 조건(삼항) 표현식 -- Swift의 "조건 ? A : B"와 완전히 같은 자리, 순서만 다름.
    return 1 if weighted_sum > threshold else 0
    # [AI 용어] 가중합이 threshold(문턱값)를 넘으면 1(=뉴런이 "발화"함), 아니면 0(=발화 안 함).
    # 이렇게 "넘으면 1, 아니면 0"으로 딱 끊어지는 함수를 계단 함수(step function)라고 부름.


if __name__ == "__main__":
    # AND 게이트를 흉내내는 뉴런: 둘 다 1일 때만 출력 1
    # [문법] [1, 1]은 리스트(list) 리터럴 -- Swift의 [1, 1] 배열 리터럴과 똑같이 생김.
    weights = [1, 1]
    threshold = 1.5

    # [문법] (0, 0) 같은 괄호 값은 튜플(tuple) -- 개수가 고정된, 서로 다른 값을 하나로 묶는 것.
    # Swift의 (Int, Int) 튜플과 같은 개념.
    test_cases = [(0, 0), (0, 1), (1, 0), (1, 1)]
    # [AI 용어] 이 4가지가 AND 게이트의 "모든 가능한 입력 조합" -- 뉴런이 이 4가지 각각에 대해
    # 실제 AND 연산과 같은 답을 내는지 확인하는 것.

    # [문법] "for a, b in test_cases:"는 튜플 언패킹(unpacking) -- 리스트 안의 (0,0) 같은 튜플을
    # 자동으로 a=0, b=0 처럼 두 변수에 나눠 담아줌. Swift의 "for (a, b) in testCases"와 같음.
    for a, b in test_cases:
        result = neuron([a, b], weights, threshold)
        # [문법] f-string 안 {a*1 + b*1}처럼 {} 안에는 변수뿐 아니라 계산식도 그대로 넣을 수 있음.
        print(f"입력=({a},{b}) -> 가중합={a*1 + b*1} -> 출력={result}")
