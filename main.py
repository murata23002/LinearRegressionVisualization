import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# データ生成
np.random.seed(0)
x = np.linspace(0, 10, 100)
y = 2.5 * x + 5 + np.random.normal(0, 2, 100)

# 初期パラメータ
w = 0.0
b = 0.0
learning_rate = 0.01
epochs = 100

# パラメータと勾配の記録
w_history = []
b_history = []
w_grad_history = []
b_grad_history = []

# 勾配降下法による最適化
for epoch in range(epochs):
    y_pred = w * x + b
    error = y_pred - y

    # 勾配計算
    w_grad = (2 / len(x)) * np.sum(error * x)
    b_grad = (2 / len(x)) * np.sum(error)

    # パラメータ更新
    w -= learning_rate * w_grad
    b -= learning_rate * b_grad

    # 記録
    w_history.append(w)
    b_history.append(b)
    w_grad_history.append(w_grad)
    b_grad_history.append(b_grad)

# グラフの準備
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(x, y, label='Input Data', color='orange')  # 入力データ

# 初期ラインと学習後のラインを追加
initial_line, = ax.plot(x, w_history[0] * x + b_history[0], 'grey', linestyle='--', label='Initial Model')  # 初期ライン
final_line, = ax.plot(x, w_history[-1] * x + b_history[-1], 'green', linestyle='--', label='Final Model')  # 最終ライン

line, = ax.plot([], [], label='Model Prediction', color='blue')  # 回帰直線
point_pred, = ax.plot([], [], 'go', label='Predicted Point')  # 予測値ポイント
point_actual, = ax.plot([], [], 'ro', label='Actual Point')  # 実測値ポイント

# 数式と値の表示
text_formula = ax.text(0.02, 0.95, '', transform=ax.transAxes)  # 関数表示
text_values = ax.text(0.02, 0.90, '', transform=ax.transAxes)  # 値の表示
text_grad = ax.text(0.02, 0.85, '', transform=ax.transAxes)  # 勾配表示
text_lr = ax.text(0.02, 0.80, '', transform=ax.transAxes)  # 学習率表示
text_epoch = ax.text(0.02, 0.75, '', transform=ax.transAxes)  # エポック表示

# 軸ラベルとタイトル
ax.set_title('Linear Regression Model Fitting')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()

# アニメーションの初期化
def init():
    line.set_data([], [])
    point_pred.set_data([], [])
    point_actual.set_data([], [])
    text_formula.set_text('')
    text_values.set_text('')
    text_grad.set_text('')
    text_lr.set_text('')
    text_epoch.set_text('')
    return line, point_pred, point_actual, text_formula, text_values, text_grad, text_lr, text_epoch

# アニメーションの更新関数
def update(frame):
    # 現在のパラメータと勾配を取得
    w = w_history[frame]
    b = b_history[frame]
    y_pred = w * x + b
    w_grad = w_grad_history[frame]
    b_grad = b_grad_history[frame]

    # モデル予測ラインの更新
    line.set_data(x, y_pred)

    # 現在の X の位置における予測値と実測値のプロット
    point_pred.set_data([x[frame]], [y_pred[frame]])  # 予測値
    point_actual.set_data([x[frame]], [y[frame]])  # 実測値

    # 数式と予測値・実測値、勾配、学習率、エポックの表示
    text_formula.set_text(f'f(x) = {w:.2f} * x + {b:.2f}')
    text_values.set_text(f'Predicted: {y_pred[frame]:.2f}, Actual: {y[frame]:.2f}')
    text_grad.set_text(f'Gradients: w_grad={w_grad:.4f}, b_grad={b_grad:.4f}')
    text_lr.set_text(f'Learning Rate: {learning_rate}')
    text_epoch.set_text(f'Epoch: {frame + 1}/{epochs}')

    return line, point_pred, point_actual, text_formula, text_values, text_grad, text_lr, text_epoch

# アニメーション実行
ani = FuncAnimation(fig, update, frames=epochs, init_func=init, blit=True, interval=100)

# 保存（必要ならば）
ani.save('linear_regression_with_gradients_and_lr_epochs.mp4', writer='ffmpeg')

# 表示
plt.show()
