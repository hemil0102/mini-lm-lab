"""
Stage 2 -- 뉴런 여러 개 = 층(layer)
1단계 뉴런 하나를, numpy로 "여러 개를 한 번에" 계산하도록 확장한다.
책 44쪽 그림의 신경망 1층/중간층/출력층이 바로 이 구조다.
"""
# [문법] import numpy as np : numpy라는 외부 라이브러리를 가져오면서 "np"라는 별명을 붙임.
# Swift의 "import numpy"에는 별명 붙이는 문법이 없는데(그냥 NumpyModule.something 식),
# 파이썬은 관례적으로 "as 별명"을 붙여서 매번 numpy.array 대신 np.array로 짧게 씀.
import numpy as np


def layer(inputs, weights, thresholds):
    """
    inputs:     shape (입력개수,)
    weights:    shape (입력개수, 뉴런개수) -- weights[:, j] 는 j번째 뉴런의 가중치
    thresholds: shape (뉴런개수,)          -- 뉴런마다 다른 기준값 가능
    반환:       shape (뉴런개수,)          -- 뉴런마다 0 또는 1
    """
    # [문법] 함수 안 첫 줄에 있는 이 세 겹 따옴표 문자열은 "함수 docstring" --
    # 이 함수가 받는 값/돌려주는 값을 설명하는 공식 주석 자리 (Swift는 이런 자리 대신 보통
    # 함수 위에 "/// 설명" 문서화 주석을 씀 -- 위치만 다르고 역할은 비슷).
    # [문법] weights[:, j] 의 ":"는 "이 축은 전부 다"라는 뜻의 슬라이싱(slicing) 문법.
    # 즉 weights[:, 0]은 "모든 행의, 0번째 열" = 0번째 뉴런에 연결된 가중치들만 쏙 뽑는 것.

    # [AI 용어] inputs @ weights : 모든 뉴런의 "가중합"을 행렬곱 한 번으로 계산.
    # [문법] "@"는 파이썬의 행렬곱 전용 연산자 -- Swift에는 이런 연산자가 없어서
    # 보통 반복문을 직접 돌리거나 별도 라이브러리 함수를 불러야 함.
    #
    # [예시] inputs=[1, 0], weights=[[1,1],[1,1]] (AND뉴런/OR뉴런용 가중치 2줄)이면:
    #   AND뉴런 열(첫 번째 열)=[1,1] -> 1*1 + 0*1 = 1
    #   OR뉴런  열(두 번째 열)=[1,1] -> 1*1 + 0*1 = 1
    #   즉 weighted_sums = [1, 1] -- 뉴런마다 따로 계산한 것과 결과가 완전히 같고, 한 번에 나온다는 점만 다름.
    weighted_sums = inputs @ weights
    # [AI 용어] 뉴런마다 자기 threshold와 비교 -- 벡터 전체에 한 번에 적용됨(브로드캐스팅, broadcasting).
    # 브로드캐스팅 = "숫자 하나짜리 규칙을 배열 전체 각 칸에 자동으로 적용해주는 numpy 기능" --
    # 반복문(for) 없이 weighted_sums의 모든 원소를 thresholds의 같은 위치 원소와 한 번에 비교해줌.
    # [문법] ".astype(int)"는 numpy 배열의 자료형(dtype)을 바꾸는 메서드 -- True/False(불리언)를
    # 1/0(정수)로 바꿔줌. Swift로 치면 Bool을 Int(bool)로 캐스팅하는 것과 비슷한 역할.
    #
    # [예시] weighted_sums=[1, 1], thresholds=[1.5, 0.5] 이면:
    #   1 > 1.5 -> False (AND뉴런: 문턱을 못 넘음)
    #   1 > 0.5 -> True  (OR뉴런: 문턱을 넘음)
    #   -> [False, True] -> .astype(int) -> [0, 1]  (반복문 없이 두 비교가 한 번에 처리됨 = 브로드캐스팅)
    return (weighted_sums > thresholds).astype(int)


if __name__ == "__main__":
    # 1단계에서 따로 만들었던 AND(threshold=1.5)와 OR(threshold=0.5) 뉴런을
    # "같은 입력을 보는 한 층"으로 합쳐서 한 번에 계산해본다.
    # [문법] np.array([[...], [...]]) : 리스트 안에 리스트를 넣으면 2차원 배열(행렬)이 됨.
    # 바깥 리스트의 원소 하나하나가 "행(row)" 하나.
    weights = np.array([
        [1, 1],  # 입력1 -> [AND뉴런 가중치, OR뉴런 가중치]
        [1, 1],  # 입력2 -> [AND뉴런 가중치, OR뉴런 가중치]
    ])
    thresholds = np.array([1.5, 0.5])
    # [AI 용어] 이 weights/thresholds 두 개가 "AND 뉴런"과 "OR 뉴런"을 하나의 층(layer)으로
    # 합쳐놓은 것 -- 같은 입력 (a, b)을 보고 두 뉴런이 동시에 서로 다른 판단을 내림.

    # [문법] "\t"는 탭(tab) 문자 -- Swift 문자열에서도 똑같이 씀. 출력 칸을 맞추려는 용도.
    print("입력\t\tAND뉴런\tOR뉴런")
    for a, b in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        out = layer(np.array([a, b]), weights, thresholds)
        print(f"({a},{b})\t\t{out[0]}\t{out[1]}")

    print()
    print("-- 여러 층 쌓기: 은닉층 3개짜리 신경망 (전부 직접 정한 가중치, 손 계산 가능) --")
    # [AI 용어] 은닉층(hidden layer) = 입력층도 출력층도 아닌, 중간에서 "미리 판단"해주는 층.
    # 은닉층 뉴런 3개: [AND뉴런, OR뉴런, "숙제만 보는" 뉴런]
    w1 = np.array([
        [1, 1, 1],  # 입력1(숙제) -> [AND, OR, 숙제전용]
        [1, 1, 0],  # 입력2(청소) -> [AND, OR, 숙제전용] (숙제전용은 청소를 아예 안 봄)
    ])
    t1 = np.array([1.5, 0.5, 0.5])

    # [AI 용어] 출력층(output layer) 뉴런 1개: 은닉층 3개 중 2개 이상 켜져야 최종 허락.
    # 은닉층의 "출력"이 이 뉴런의 "입력"이 되는 것 -- 이게 책에서 말한 토폴로지(연결 구조)의 실제 모습.
    w2 = np.array([[1], [1], [1]])
    t2 = np.array([1.5])

    for a, b in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        inputs = np.array([a, b])
        hidden = layer(inputs, w1, t1)  # [AI 용어] 순전파(forward pass)의 첫 단계 -- 입력을 은닉층에 통과시킴
        output = layer(hidden, w2, t2)  # [AI 용어] 은닉층 출력을 다시 출력층에 통과시킴 -- 순전파의 마지막 단계
        print(f"입력=({a},{b}) -> 은닉층 출력={hidden} -> 최종 출력={output[0]}")
