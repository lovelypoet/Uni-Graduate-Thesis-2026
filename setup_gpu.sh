#!/bin/bash
# setup_gpu.sh — Fix "Cannot dlopen GPU libraries" cho TensorFlow trên WSL.
# GPU (RTX 4060) đã thấy qua nvidia-smi, chỉ cần trỏ TF tới thư viện CUDA do pip cài.
#
# Cách dùng:
#   source setup_gpu.sh
# (phải dùng "source", KHÔNG chạy "./setup_gpu.sh" — để biến môi trường giữ lại)

# Tìm thư mục gốc của các package nvidia-*-cu12 mà pip đã cài
NV_BASE=$(python3 -c "import os, nvidia; print(os.path.dirname(nvidia.__file__))")

# Gom tất cả thư mục */lib bên trong (cudnn, cublas, cufft, ...) vào LD_LIBRARY_PATH
CUDA_LIBS=$(python3 -c "
import os, nvidia
base = os.path.dirname(nvidia.__file__)
paths = []
for d in os.listdir(base):
    lib = os.path.join(base, d, 'lib')
    if os.path.isdir(lib):
        paths.append(lib)
print(':'.join(paths))
")

export LD_LIBRARY_PATH=$CUDA_LIBS:$LD_LIBRARY_PATH

echo "✅ Đã set LD_LIBRARY_PATH:"
echo "$CUDA_LIBS" | tr ':' '\n' | sed 's/^/   /'
echo ""
echo "Kiểm tra GPU:"
python3 -c "import tensorflow as tf; gpus=tf.config.list_physical_devices('GPU'); print('GPU thấy:', gpus if gpus else 'KHÔNG — xem hướng dẫn bên dưới')" 2>/dev/null