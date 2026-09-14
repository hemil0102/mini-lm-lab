def predict(x, w):
    return w * x

def loss(pred, target):
    return (pred - target) ** 2

def gradient(x, pred, target):
    # 손실 (w*x - target)^2 을 w에 대해 미분하면 2*(pred-target)*x 가 된다.
    # (미분 과정은 생략 — 결과 공식만 사용)
    return 2 * (pred - target) * x

if __name__ == "__main__":
    x = 2
    target = 10
    w = 1.0
    learning_rate = 0.01

    for step in range(15):
        pred = predict(x, w)
        current_loss = loss(pred, target)
        grad = gradient(x, pred, target)
        print(f"step={step:2d} w={w:.4f} 예측={pred:.4f} 손실={current_loss:.4f} 기울기={grad:.4f}")
        w = w - learning_rate * grad

    print(f"\n최종 w={w:.4f}  (정답: w=5일 때 예측={x*5}=목표값 10)")
