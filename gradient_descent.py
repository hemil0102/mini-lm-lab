# "#"으로 시작하는 줄은 주석 — Swift의 "//"랑 같은 역할, 실행에 영향 없음.

def predict(x, w):
    # def = 함수 정의 키워드 (Swift의 func와 같은 자리).
    # 파이썬은 매개변수에 타입을 안 써도 됨 (Swift: func predict(x: Double, w: Double) -> Double 처럼
    # 타입을 꼭 적어야 하는 것과 다름 — x, w가 무슨 타입이든 일단 받고, 곱셈이 안 되는 타입이 들어오면
    # 실행할 때(런타임에) 에러가 남. "동적 타이핑"이라고 부름).
    return w * x
    # return = Swift와 동일, 값을 함수 밖으로 돌려줌.

def loss(pred, target):
    return (pred - target) ** 2
    # ** = 거듭제곱 연산자 (Swift엔 이런 연산자가 없어서 pow(pred - target, 2)처럼 함수로 씀).

def gradient(x, pred, target):
    # 손실 (w*x - target)^2 을 w에 대해 미분하면 2*(pred-target)*x 가 된다.
    # (미분 과정은 생략 — 결과 공식만 사용)
    return 2 * (pred - target) * x

if __name__ == "__main__":
    # 파이썬 특유의 관용구. 이 파일을 "직접 실행"했을 때만 아래 코드가 도는데,
    # 다른 파일에서 이 파일을 "가져다 쓸(import)" 때는 안 도는 게 의도임.
    # Swift는 main.swift나 @main이 자동으로 이 역할을 해줘서 이런 줄이 따로 필요 없는데,
    # 파이썬은 모든 .py 파일이 "직접 실행도, 부품으로 가져다 쓰기도" 둘 다 가능하다 보니
    # "지금 이 파일이 시작점으로 실행된 게 맞나?"를 직접 확인해주는 관례가 생긴 것.
    x = 2
    target = 10
    w = 1.0
    learning_rate = 0.01

    for step in range(15):
        # range(15) = 0부터 14까지 15개의 정수를 순서대로 만들어주는 것 (Swift의 0..<15와 같음).
        # for step in ... : 만들어진 값을 하나씩 step에 넣으면서 아래 블록을 반복 (Swift의 for step in 0..<15와 동일한 개념).
        pred = predict(x, w)
        current_loss = loss(pred, target)
        grad = gradient(x, pred, target)

        print(f"step={step:2d} w={w:.4f} 예측={pred:.4f} 손실={current_loss:.4f} 기울기={grad:.4f}")
        # f"..." = f-string. 문자열 안에 {변수}를 넣으면 그 값을 바로 끼워 넣어줌
        # (Swift의 "step=\(step)"과 똑같은 역할, f-string은 {} 안에 \만 없다고 보면 됨).
        # {step:2d} 의 ":2d"는 "정수를 최소 2자리로, 자리가 모자라면 공백으로 채워서" 표시하라는 서식 지정.
        # {w:.4f} 의 ":.4f"는 "소수점 아래 4자리까지만 표시"하라는 뜻 (Swift의 String(format: "%.4f", w)와 같은 용도).

        w = w - learning_rate * grad

    print(f"\n최종 w={w:.4f}  (정답: w=5일 때 예측={x*5}=목표값 10)")
    # 문자열 맨 앞의 "\n"은 줄바꿈 문자 — Swift와 동일.
