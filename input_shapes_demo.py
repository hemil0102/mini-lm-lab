"""
보충 자료 -- 책 45쪽 "문제 입력" 시각화
왜 시각 패턴은 2차원 배열이고, 청각 패턴도 2차원 배열인지 실제로 그려서 확인한다.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 화면 없이 파일로만 저장
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Apple SD Gothic Neo"  # macOS 한글 지원 폰트
plt.rcParams["axes.unicode_minus"] = False

# ---- 1. 이미지: 진짜로 "숫자 배열 = 픽셀 밝기"인지 확인 ----
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
# 세로축=주파수 성분(낮음->높음), 가로축=시간. 시간이 지나며 저음->고음으로 올라가는 소리를 흉내냄
time_steps, freq_bins = 12, 6
spectrogram = np.zeros((freq_bins, time_steps))
for t in range(time_steps):
    f = int(t / time_steps * freq_bins)
    spectrogram[f, t] = 1.0
    if f + 1 < freq_bins:
        spectrogram[f + 1, t] = 0.4  # 살짝 번지는 느낌

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

ax = axes[0]
ax.imshow(image, cmap="Greys", vmin=0, vmax=1)
for r in range(image.shape[0]):
    for c in range(image.shape[1]):
        ax.text(c, r, str(image[r, c]), ha="center", va="center",
                 color="red", fontsize=11)
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
