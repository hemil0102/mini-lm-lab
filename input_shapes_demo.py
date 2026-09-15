"""
보충 자료 -- 책 45쪽 "문제 입력" 시각화
왜 시각 패턴은 2차원 배열이고, 청각 패턴도 2차원 배열인지 실제로 그려서 확인한다.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 화면 없이 파일로만 저장
# [문법] matplotlib.use("Agg") 처럼 괄호 안에 값을 하나 넣어 부르는 건 그냥 보통 함수 호출.
# "Agg"는 "화면에 창을 띄우지 않고 그림을 파일로만 그리는 모드" 이름 -- 서버/터미널 환경용 설정.
import matplotlib.pyplot as plt
# [문법] plt.rcParams[...] = ... 는 도서관(matplotlib) 전체의 기본 설정값을 딕셔너리처럼
# 대괄호로 꺼내서 바꾸는 것. Swift에는 이런 "전역 설정을 딕셔너리 문법으로 바꾸는" 관례는 잘 없음.
plt.rcParams["font.family"] = "Apple SD Gothic Neo"  # macOS 한글 지원 폰트
plt.rcParams["axes.unicode_minus"] = False

# ---- 1. 이미지: 진짜로 "숫자 배열 = 픽셀 밝기"인지 확인 ----
# [AI 용어] 아래 image 배열이 바로 "시각 패턴 입력" -- 각 숫자가 그 위치 픽셀의 밝기(0=흰색, 1=검정).
# 6x6 흑백 이미지 -- 숫자가 클수록 진하게(검게) 칠해진다
image = np.array([
    [0, 0, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 0],
    [1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1],
    [0, 0, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 0],
])

# ---- 2. 소리: 스펙트로그램(주파수 x 시간) 예시 ----
# [AI 용어] spectrogram = "청각 패턴 입력" -- 원래 1차원인 소리 신호를 "이 시간에 이 주파수가
# 얼마나 있는지"로 잘라서 2차원 배열로 바꾼 것.
# 세로축=주파수 성분(낮음->높음), 가로축=시간. 시간이 지나며 저음->고음으로 올라가는 소리를 흉내냄
time_steps, freq_bins = 12, 6
# [문법] np.zeros((freq_bins, time_steps)) 의 괄호 두 겹 -- 바깥 ()는 함수 호출,
# 안쪽 (freq_bins, time_steps)는 튜플 하나로 묶은 "배열의 모양(shape)"을 넘기는 것.
# 즉 "0으로 채워진 6행 x 12열짜리 배열을 만들어라"는 뜻.
spectrogram = np.zeros((freq_bins, time_steps))
for t in range(time_steps):
    f = int(t / time_steps * freq_bins)
    # [문법] int(...) 는 실수를 정수로 변환(형변환) -- Swift의 Int(someDouble)과 같은 역할.
    #
    # [예시] time_steps=12, freq_bins=6일 때 t=0이면: 0/12*6=0.0 -> int(0.0)=0 (맨 아래 주파수 칸)
    #   t=6이면: 6/12*6=3.0 -> int(3.0)=3 (중간 칸). t=11이면: 11/12*6=5.5 -> int(5.5)=5 (맨 위 칸)
    #   -> 시간(t)이 흐를수록 f가 0에서 5까지 점점 커짐 = "시간이 지나며 저음에서 고음으로 올라가는" 효과.
    spectrogram[f, t] = 1.0
    if f + 1 < freq_bins:
        spectrogram[f + 1, t] = 0.4  # 살짝 번지는 느낌

# [문법] fig, axes = plt.subplots(1, 2, figsize=(11, 5)) 처럼 왼쪽에 변수 두 개를 콤마로
# 나열해서 받는 것도 튜플 언패킹 -- subplots()가 (그림 전체, 서브플롯 배열) 두 값을 튜플로
# 돌려주는데, 그걸 fig / axes 두 변수에 바로 나눠 담는 것.
# [문법] figsize=(11, 5) 처럼 "이름=값" 형태로 넘기는 걸 키워드 인자(keyword argument)라고 함 --
# Swift에서 함수 부를 때 "label: value" 쓰는 것과 거의 같은 느낌.
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

ax = axes[0]  # [문법] axes가 배열이라 [0], [1]로 첫 번째/두 번째 그래프 칸에 접근
ax.imshow(image, cmap="Greys", vmin=0, vmax=1)
for r in range(image.shape[0]):
    # [문법] image.shape 는 numpy 배열의 "속성(attribute)" -- (행 개수, 열 개수) 튜플을 담고 있음.
    # image.shape[0]=행 개수, image.shape[1]=열 개수. 함수가 아니라 값이라 괄호 없이 접근.
    for c in range(image.shape[1]):
        ax.text(c, r, str(image[r, c]), ha="center", va="center",
                 color="red", fontsize=11)
        # [문법] ha=, va=, color=, fontsize= 도 전부 키워드 인자 -- 순서 상관없이
        # "이 값은 이 이름의 매개변수다"라고 콕 집어서 넘기는 방식.
ax.set_xticks(range(image.shape[1]))
ax.set_yticks(range(image.shape[0]))
ax.set_xlabel("가로 위치 (열)")
ax.set_ylabel("세로 위치 (행)")
ax.set_title("이미지 = 2차원 배열\n(숫자 하나하나가 픽셀 한 칸의 밝기)")

ax = axes[1]
im = ax.imshow(spectrogram, cmap="magma", aspect="auto", origin="lower")
ax.set_xlabel("시간 -->")
ax.set_ylabel("주파수 성분 (낮음 -> 높음)")
ax.set_title("소리(스펙트로그램) = 2차원 배열\n(세로=주파수, 가로=시간)")
fig.colorbar(im, ax=ax, label="에너지 크기")

fig.suptitle("책 45쪽 '문제 입력' -- 배열 모양은 데이터의 자연스러운 구조를 따른다", fontsize=12)
fig.tight_layout()
fig.savefig("input_shapes_demo.png", dpi=150)
print("이미지 배열:")
print(image)
print("saved: input_shapes_demo.png")
