import torch
import nvdiffrast.torch as dr
import os


def precompile():
    print("开始预编译 nvdiffrast kernels...")
    
    # 确保有 CUDA 环境 (构建机器必须安装了 NVCC 和 CUDA 驱动)
    if not torch.cuda.is_available():
        print("警告: 未检测到 CUDA 设备，无法进行预编译！")
        return
    device = torch.device("cuda")
    
    # 1. 触发 RasterizeCudaContext 编译 (这是最常用的，纯 CUDA 版本)
    try:
        print("Compiling RasterizeCudaContext...")
        # 这一步会触发 torch.utils.cpp_extension.load 编译 .so 文件
        ctx = dr.RasterizeCudaContext(device=device)
        print("RasterizeCudaContext 编译完成。")
    except Exception as e:
        print(f"RasterizeCudaContext 编译失败: {e}")
    # 2. (可选) 如果你需要用 GL Context，取消下面注释
    # try:
    #     print("Compiling RasterizeGLContext...")
    #     ctx_gl = dr.RasterizeGLContext(device=device)
    #     print("RasterizeGLContext 编译完成。")
    # except Exception as e:
    #     print(f"RasterizeGLContext 编译失败 (可能缺少 OpenGL 库): {e}")
    print("所有内核已缓存至 PyTorch 扩展目录。")

if __name__ == "__main__":
    os.environ['TORCH_CUDA_ARCH_LIST'] = "8.6;8.9;9.0"
    precompile()