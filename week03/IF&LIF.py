import numpy as np
import matplotlib.pyplot as plt

# 1. Khai báo các siêu tham số sinh học và cấu hình mô phỏng
T = 100.0          # Tổng thời gian mô phỏng (ms)
dt = 0.1           # Bước thời gian rời rạc hóa (ms)
time = np.arange(0, T, dt) # Vector thời gian

tau_m = 10.0       # Hằng số thời gian màng (ms)
V_rest = -65.0     # Điện thế nghỉ tĩnh (mV)
V_th = -50.0       # Ngưỡng kích hoạt phát xung (mV)
V_reset = -65.0    # Điện thế trạng thái thiết lập lại (mV)
R = 1.0            # Trở kháng màng (đơn vị quy chuẩn)

# 2. Xây dựng hàm kích thích đầu vào I(t)
# Thiết lập xung vuông cường độ 20 đơn vị trong khoảng thời gian [20, 80] ms
I = np.zeros(len(time))
I[int(20/dt) : int(80/dt)] = 20.0 

# 3. Khởi tạo mảng lưu trữ biến trạng thái
V = np.zeros(len(time))
V[0] = V_rest      # Điều kiện ban đầu V(0) = V_rest
spike_times = []   # Mảng ghi nhận tọa độ thời gian của các xung

# 4. Tích phân số học theo phương pháp Euler
for i in range(1, len(time)):
    # Tính toán gradient điện thế theo phương trình LIF
    dV_dt = (-(V[i-1] - V_rest) + R * I[i]) / tau_m
    
    # Cập nhật trạng thái V(t)
    V[i] = V[i-1] + dV_dt * dt
    
    # Hàm kích hoạt bước nhảy (Heaviside step function logic) cho Spiking
    if V[i] >= V_th:
        V[i] = V_reset                # Thực thi cơ chế reset
        spike_times.append(time[i])   # Lưu trữ thời điểm phát xung

# 5. Trực quan hóa dữ liệu (Data Visualization)
plt.figure(figsize=(10, 6))

# Đồ thị kích thích đầu vào
plt.subplot(2, 1, 1)
plt.plot(time, I, color='orange', linewidth=2)
plt.title("Đồ thị dòng điện kích thích đầu vào $I(t)$")
plt.ylabel("Cường độ $I$")
plt.grid(True, alpha=0.3)

# Đồ thị phản hồi điện thế màng
plt.subplot(2, 1, 2)
plt.plot(time, V, color='blue', linewidth=2)
plt.axhline(y=V_th, color='red', linestyle='--', label=f'Ngưỡng kích hoạt ($V_{{th}}$ = {V_th}mV)')
plt.axhline(y=V_rest, color='green', linestyle=':', label=f'Điện thế nghỉ ($V_{{rest}}$ = {V_rest}mV)')

# Trực quan hóa các sự kiện phát xung (Raster plot overlay)
for t_spike in spike_times:
    plt.axvline(x=t_spike, color='purple', linestyle='-', alpha=0.5)

plt.title("Động lực học điện thế màng $V(t)$ của mô hình LIF")
plt.xlabel("Thời gian (ms)")
plt.ylabel("Điện thế màng (mV)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Tổng số xung kích hoạt trong toàn chu kỳ: {len(spike_times)}")