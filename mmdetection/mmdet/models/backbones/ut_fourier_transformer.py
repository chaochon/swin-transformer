import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt

from torchvision.io import read_image

def process_image_torch(image_path, low_freq_size=1):
    # 读取图像并转换为 Tensor
    img_tensor = read_image(image_path)

    # 提取通道数、宽度和高度
    C, H, W = img_tensor.shape

    # 初始化结果张量
    processed_tensor = torch.zeros_like(img_tensor)

    spectrogram_tensor1 = torch.zeros_like(img_tensor)
    spectrogramList = []
    mask1 = 1
    for c in range(C):  # 对每个通道单独处理
        # 1. 对单通道进行 2D 傅里叶变换
        f = torch.fft.fft2(img_tensor[c])  # 2D FFT
        fshift = torch.fft.fftshift(f)  # 中心化
        # 2. 构造低频矩形掩膜
        mask = torch.ones(H, W, dtype=torch.complex64, device=f.device)
        crow, ccol = H // 2, W // 2
        print("H: {}, W: {}, low_freq_size: {}".format(H, W, low_freq_size))
        mask[crow - low_freq_size:crow + low_freq_size, ccol - low_freq_size:ccol + low_freq_size] = 0

        # 3. 应用低频掩膜
        fshift_filtered = fshift * mask

        mask1 = mask
        #record spectrogram
        # 计算频谱的幅值
        magnitude = torch.abs(fshift)
        # 对幅值进行对数缩放
        magnitude_spectrum = torch.log(1 + magnitude)
        spectrogramList.append(magnitude_spectrum)
        spectrogram_tensor1[c] = magnitude_spectrum
        if c == C-1:
            spectrogramList.append(spectrogram_tensor1)

        # 4. 逆傅里叶变换恢复
        f_ishift = torch.fft.ifftshift(fshift_filtered)  # 中心化还原
        img_reconstructed = torch.fft.ifft2(f_ishift)  # 2D IFFT
        img_reconstructed = img_reconstructed.abs()  # 取实部幅值

        # 保存处理后的通道
        processed_tensor[c] = img_reconstructed

    # 将结果限制在 [0, 255]
    processed_tensor = torch.clamp(processed_tensor, 0, 255)

    # 可视化结果
    original_img = T.ToPILImage()(img_tensor)
    processed_img = T.ToPILImage()(processed_tensor)

    plot_tensor_histogram(processed_tensor)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(original_img)
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.title("Mask")
    plt.imshow(T.ToPILImage()(mask1.abs()))
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Reconstructed Image")
    plt.imshow(processed_img)
    plt.axis("off")

    plt.figure(figsize=(12, 6))
    for i, spectrogram in enumerate(spectrogramList):
        plt.subplot(2, 2, i+1)
        plt.title("spectrogram {}".format(i+1))
        if i == len(spectrogramList)-1:
            plt.imshow(spectrogram.permute(1,2,0).numpy())
        else:
            plt.imshow(spectrogram.numpy(), cmap='gray')
        plt.axis("off")

    plt.figure(figsize=(12, 6))
    rgb = ['red', 'green', 'blue']
    for i in range(C):
        plt.subplot(1, 3, i+1)
        plt.title("{}".format(rgb[i]))
        plt.imshow(T.ToPILImage()(img_tensor[i]), cmap='gray')
        plt.axis("off")

    plt.show()

    return processed_tensor

def plot_tensor_histogram(tensor, bins=50):
    """
    绘制张量的直方图。
    :param tensor: 输入 PyTorch 张量，形状为 (C, H, W) 或 (H, W)。
    :param bins: 直方图分区数。
    """
    # 将张量展平为一维
    flattened_tensor = tensor.flatten()

    # 统计直方图数据
    min_val, max_val = flattened_tensor.min().item(), flattened_tensor.max().item()
    hist = torch.histc(flattened_tensor.to(torch.float32), bins=bins, min=min_val, max=max_val)

    # 创建对应的 x 轴范围
    bin_edges = torch.linspace(min_val, max_val, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # 绘制直方图
    plt.figure(figsize=(8, 5))
    plt.bar(bin_centers.numpy(), hist.numpy(), width=(max_val - min_val) / bins, align="center")
    plt.title("Histogram of Tensor Values")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()


# 调用函数处理图像
image_path = "/home/caochong/experiments/datasets/VOC0712/VOCdevkit/VOC2007/JPEGImages/003522.jpg"  # 替换为你的图像路径
reconstructed_img = process_image_torch(image_path)
