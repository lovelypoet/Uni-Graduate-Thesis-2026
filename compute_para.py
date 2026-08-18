#!/usr/bin/env python3
"""
compute_params_flops.py — Tính chính xác Params + FLOPs cho bảng
tab:deployment trong thesis (Table~\\ref{tab:deployment}).

Vì sao cần script riêng thay vì dùng số trong paper gốc (EfficientNet-B0
~5.3M params, MobileNetV3-Large ~5.4M params): số trong paper là cho
backbone TRẦN với classifier ImageNet 1000-lớp gốc (include_top=True),
KHÔNG phải model thật đang dùng trong pipeline này (include_top=False +
head tùy chỉnh nhỏ). Chênh lệch không hề nhỏ (model thật ~3.0M so với
~5.4M của kiến trúc gốc) nên PHẢI tính trực tiếp trên model đã lưu,
không được lấy số trong paper.

QUAN TRỌNG — sửa so với bản trước: bản trước thử keras-flops trước rồi
mới fallback sang tf profiler nếu lỗi (vd HardSwish của MobileNetV3
không được keras-flops hỗ trợ). Vấn đề: 2 phương pháp đó đếm FLOPs theo
quy ước khác nhau, nên nếu Teacher đi nhánh A còn Student đi nhánh B,
CỘT FLOPs TRONG BẢNG SẼ SO SÁNH KHẬP KHIỄNG giữa các hàng — đúng loại
lỗi "confound" cần tránh khi so sánh nhiều model trong 1 bảng. Bản này
dùng DUY NHẤT MỘT phương pháp (TF profiler) cho MỌI model, đảm bảo công
bằng tuyệt đối giữa các hàng, và ẩn báo cáo profiler dài dằng dặc mà
bản trước để nó tự in ra làm ngợp terminal.

Yêu cầu: chạy trên máy đã có TensorFlow, KHÔNG cần GPU cho việc này.
    python3 compute_params_flops.py 2>&1 | tee params_flops.log
(dùng tee để lưu lại toàn bộ output, tránh bị cuộn mất trên terminal
như lần chạy trước.)
"""
import gc
import tensorflow as tf
import config as C

MODELS = {
    "Teacher (EfficientNet-B0)": C.TEACHER_PATH,
    "Student no-KD": C.STUDENT_NOKD_PATH,
    "Student KD": C.STUDENT_KD_PATH,
}


def count_params(model):
    total = model.count_params()
    trainable = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    return total, trainable


def compute_flops(model, input_shape):
    """FLOPs cho 1 forward pass, dung DUY NHAT tf profiler cho MOI model
    (khong thu keras-flops truoc) de dam bao cung 1 quy uoc dem across
    toan bo bang -- neu doi phuong phap giua cac hang, cot FLOPs se
    khong so sanh duoc voi nhau nua.

    output bi an (with_empty_output) de khong in bao cao dai ra
    terminal; ham tra ve dung 1 so duy nhat: total_float_ops (dem moi
    phep nhan/cong la 1 "flop", KHONG chia 2 theo quy uoc MAC -- ghi ro
    quy uoc nay trong caption bang deployment cua thesis de nguoi doc
    biet don vi)."""
    concrete = tf.function(lambda x: model(x)).get_concrete_function(
        tf.TensorSpec(input_shape, model.inputs[0].dtype))

    opts = (
        tf.compat.v1.profiler.ProfileOptionBuilder(
            tf.compat.v1.profiler.ProfileOptionBuilder.float_operation())
        .with_empty_output()   # <- an bao cao dai, chi lay so tong
        .build()
    )
    graph_info = tf.compat.v1.profiler.profile(
        graph=concrete.graph, options=opts)
    return graph_info.total_float_ops


def main():
    print("=" * 70)
    print("PARAMS + FLOPs cho Table~\\ref{tab:deployment}")
    print("Phương pháp FLOPs: tf.compat.v1.profiler, DÙNG CHUNG cho mọi")
    print("model (đảm bảo công bằng giữa các hàng trong bảng).")
    print("=" * 70)
    print(f"{'Model':<30} {'Params (total)':>15} {'Params (M)':>12} {'GFLOPs':>10}")
    print("-" * 70)

    input_shape = (1, C.IMG_SIZE, C.IMG_SIZE, 3)
    rows = []

    for name, path in MODELS.items():
        if not path.exists():
            print(f"{name:<30} [không tìm thấy file -- kiểm tra "
                  f"TEACHER_PATH/STUDENT_NOKD_PATH/STUDENT_KD_PATH trong config.py]")
            rows.append((name, None, None))
            continue

        model = tf.keras.models.load_model(path, compile=False)
        total, _ = count_params(model)

        try:
            flops = compute_flops(model, input_shape)
        except Exception as e:
            print(f"  ⚠️  {name}: FLOPs profiler thất bại ({e}) — GFLOPs để trống,"
                  f" params vẫn dùng được.")
            flops = None

        flops_g = f"{flops / 1e9:.3f}" if flops else "N/A"
        print(f"{name:<30} {total:>15,} {total/1e6:>11.2f}M {flops_g:>10}")
        rows.append((name, total, flops))

        del model
        tf.keras.backend.clear_session()
        gc.collect()

    print("\nGhi chú:")
    print("- CẢ 3 HÀNG dùng CÙNG một phương pháp đếm FLOPs (tf profiler) —")
    print("  so sánh giữa các hàng là công bằng, không lệch do khác nhánh code.")
    print("- 'Params (total)' bao gồm cả backbone + custom head, dùng cho")
    print("  MỌI hàng trong bảng kể cả TFLite (kiến trúc không đổi, chỉ đổi")
    print("  định dạng lưu trữ / độ chính xác số).")
    print("- FLOPs của bản Keras và TFLite float32 là NHƯ NHAU (cùng đồ thị")
    print("  phép tính); INT8 chỉ đổi CHI PHÍ mỗi phép tính trên phần cứng hỗ")
    print("  trợ, không đổi số lượng phép tính -> dùng chung 1 cột FLOPs cho")
    print("  3 hàng Keras/float32/INT8 của cùng 1 model KD.")
    print("- Quy ước: total_float_ops đếm MỖI phép nhân VÀ mỗi phép cộng là")
    print("  1 flop riêng (không gộp MAC=1 flop). Nếu muốn so với số trong")
    print("  paper gốc (thường theo quy ước MAC*2), chia số này cho 2 trước")
    print("  khi so sánh -- nhưng KHÔNG cần chia khi so sánh nội bộ giữa các")
    print("  hàng trong Table~\\ref{tab:deployment}, vì cả 3 hàng đã cùng")
    print("  quy ước rồi.")
    print("- Copy các số Params(M) và GFLOPs vào đúng cột trong")
    print("  Table~\\ref{tab:deployment} của thesis_final.tex, thay cho")
    print("  \\textcolor{accentred}{[FILL]}.")

    missing = [r[0] for r in rows if r[1] is None]
    if missing:
        print(f"\n⚠️  CHƯA có số cho: {missing} — kiểm tra lại đường dẫn model")
        print("   trong config.py trước khi coi bảng này là đầy đủ.")


if __name__ == "__main__":
    main()