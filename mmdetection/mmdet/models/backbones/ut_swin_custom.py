import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt

from torchvision.io import read_image

from PIL import Image
import numpy as np
import torch.nn.functional as F

def process_image_torch(image_path, low_freq_size=4):
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

    #plot_tensor_histogram(processed_tensor)

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

    # plt.figure(figsize=(12, 6))
    # for i, spectrogram in enumerate(spectrogramList):
    #     plt.subplot(2, 2, i+1)
    #     plt.title("spectrogram {}".format(i+1))
    #     if i == len(spectrogramList)-1:
    #         print('No.{} shape:{}'.format(i, spectrogram.shape))
    #         plt.imshow(spectrogram.permute(1,2,0).numpy())
    #     else:
    #         print('No.{} shape:{}'.format(i, spectrogram.shape))
    #         plt.imshow(spectrogram.numpy(), cmap='gray')
    #     plt.axis("off")

    # plt.figure(figsize=(12, 6))
    # rgb = ['red', 'green', 'blue']
    # for i in range(C):
    #     plt.subplot(1, 3, i+1)
    #     plt.title("{}".format(rgb[i]))
    #     plt.imshow(T.ToPILImage()(img_tensor[i]), cmap='gray')
    #     plt.axis("off")

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

#==================================================

# def high_pass_filter(fft_tensor, cutoff_frequency):
#     """
#     Applies a high-pass filter on the image in the frequency domain.
#
#     Parameters:
#     - img_tensor: The image tensor in the frequency domain (complex).
#     - cutoff_frequency: The cutoff frequency for the high-pass filter.
#
#     Returns:
#     - filtered_img: The filtered image tensor in the frequency domain.
#     """
#     # Get the shape of the image
#     _, H, W = fft_tensor.shape
#
#     # 2. 构造低频矩形掩膜
#     mask = torch.ones(H, W, dtype=torch.complex64)
#     crow, ccol = H // 2, W // 2
#     print("H: {}, W: {}, low_freq_size: {}".format(H, W, low_freq_size))
#     mask[crow - low_freq_size:crow + low_freq_size, ccol - low_freq_size:ccol + low_freq_size] = 0
#
#     # 3. 应用低频掩膜
#     fshift_filtered = fshift * mask
#
#     # Apply the high-pass filter
#     filtered_img = img_tensor * high_pass_mask
#
#     return filtered_img


def process_gray_image(image_path, low_freq_size=4):
    # 1. Load the image and convert it to grayscale
    img = Image.open(image_path).convert('L')
    # img_np = np.array(img) / 255.0  # Normalize to [0, 1]
    img_np = np.array(img)

    # 2. Convert the image to a tensor and add a batch dimension
    img_tensor = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    _, _, H, W = img_tensor.shape

    # 3. Apply Fourier Transform (FFT) to the image
    img_fft = torch.fft.fft2(img_tensor)
    fshift = torch.fft.fftshift(img_fft)  # 中心化

    # 4. Apply high-pass filter in the frequency domain
    mask = torch.ones(H, W, dtype=torch.complex64, device=img_fft.device)
    crow, ccol = H // 2, W // 2
    print("H: {}, W: {}, low_freq_size: {}".format(H, W, low_freq_size))
    mask[crow - low_freq_size:crow + low_freq_size, ccol - low_freq_size:ccol + low_freq_size] = 0
    fshift_filtered = fshift * mask


    # 5. Apply Inverse Fourier Transform (IFFT) to get the filtered image
    f_ishift = torch.fft.ifftshift(fshift_filtered)  # 中心化还原
    img_reconstructed = torch.fft.ifft2(f_ishift)

    # 6. Get the real part of the reconstructed image and normalize it
    img_reconstructed = img_reconstructed.real.squeeze().detach().numpy()
    img_reconstructed = np.clip(img_reconstructed, 0, 255)

    # 7. Display the original and reconstructed images
    fig, ax = plt.subplots(1, 3, figsize=(12, 6))
    ax[0].imshow(img_np, cmap='gray')
    ax[0].set_title("Original Image")
    ax[0].axis('off')

    ax[1].imshow(mask.real.squeeze().detach().numpy(), cmap='gray')
    ax[1].set_title("Filtered and Reconstructed Image")
    ax[1].axis('off')

    ax[2].imshow(img_reconstructed, cmap='gray')
    ax[2].set_title("Filtered and Reconstructed Image")
    ax[2].axis('off')


    plt.show()


def gaussian_kernel(kernel_size=5, sigma=1.0):
    """
    Generates a Gaussian kernel for image smoothing.
    """
    x = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=torch.float32)
    x = x.view(-1, 1)
    y = x.t()
    kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel

def sobel_kernels():
    """
    Returns Sobel kernels for gradient computation.
    """
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
    return sobel_x, sobel_y

def non_maximum_suppression(gradient_magnitude, gradient_direction):
    """
    Performs non-maximum suppression to thin edges.
    """
    rows, cols = gradient_magnitude.shape
    suppressed = torch.zeros_like(gradient_magnitude)

    angle = gradient_direction * (180.0 / np.pi)
    angle = torch.remainder(angle, 180)  # Keep angles in [0, 180)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            q = 255  # Default high value
            r = 255
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = gradient_magnitude[i, j + 1]
                r = gradient_magnitude[i, j - 1]
            elif 22.5 <= angle[i, j] < 67.5:
                q = gradient_magnitude[i - 1, j + 1]
                r = gradient_magnitude[i + 1, j - 1]
            elif 67.5 <= angle[i, j] < 112.5:
                q = gradient_magnitude[i - 1, j]
                r = gradient_magnitude[i + 1, j]
            elif 112.5 <= angle[i, j] < 157.5:
                q = gradient_magnitude[i + 1, j + 1]
                r = gradient_magnitude[i - 1, j - 1]

            if gradient_magnitude[i, j] >= q and gradient_magnitude[i, j] >= r:
                suppressed[i, j] = gradient_magnitude[i, j]

    return suppressed

def canny_edge_detection_torch(image, low_threshold=0.1, high_threshold=0.3, sigma=1.0):
    """
    Implements Canny edge detection using PyTorch.
    """
    # Step 1: Gaussian blur
    kernel = gaussian_kernel(sigma=sigma)
    kernel = kernel.view(1, 1, *kernel.shape)  # Reshape for conv2d
    image = image.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    blurred = F.conv2d(image, kernel, padding=kernel.size(-1) // 2)

    # Step 2: Sobel gradient computation
    sobel_x, sobel_y = sobel_kernels()
    sobel_x = sobel_x.view(1, 1, *sobel_x.shape)
    sobel_y = sobel_y.view(1, 1, *sobel_y.shape)
    grad_x = F.conv2d(blurred, sobel_x, padding=1)
    grad_y = F.conv2d(blurred, sobel_y, padding=1)
    gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2).squeeze()
    gradient_direction = torch.atan2(grad_y, grad_x).squeeze()

    # Step 3: Non-maximum suppression
    suppressed = non_maximum_suppression(gradient_magnitude, gradient_direction)

    # # Step 4: Double threshold
    # strong = suppressed.max() * high_threshold
    # weak = strong * low_threshold
    # strong_edges = (suppressed >= strong)
    # weak_edges = ((suppressed >= weak) & (suppressed < strong))
    #
    # # Step 5: Edge tracking by hysteresis
    # edges = torch.zeros_like(suppressed, dtype=torch.uint8)
    # edges[strong_edges] = 255
    #
    # for i in range(1, suppressed.shape[0] - 1):
    #     for j in range(1, suppressed.shape[1] - 1):
    #         if weak_edges[i, j]:
    #             if strong_edges[i - 1:i + 2, j - 1:j + 2].any():
    #                 edges[i, j] = 255

    return suppressed * 255

def process_image_with_canny(image_path, low_threshold=0.1, high_threshold=0.3, sigma=1.0):
    # Load and preprocess the image
    img = Image.open(image_path).convert('L')
    img_np = np.array(img) / 255.0  # Normalize to [0, 1]
    img_tensor = torch.tensor(img_np, dtype=torch.float32)

    # Perform Canny edge detection
    edges = canny_edge_detection_torch(img_tensor, low_threshold, high_threshold, sigma)

    # Display results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img_np, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Canny Edges")
    plt.imshow(edges.cpu().numpy(), cmap='gray')
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    # 调用函数处理图像
    image_path = "/home/caochong/experiments/datasets/VOC0712/VOCdevkit/VOC2007/JPEGImages/003523.jpg"  # 替换为你的图像路径
    lena_image_path = "/home/caochong/experiments/swin-transformer/temp/lena.jpg"
    # for i in range(100):
    #     reconstructed_img = process_gray_image(image_path, low_freq_size = i)
    # reconstructed_img = process_gray_image(image_path, low_freq_size = 30)
    process_image_with_canny(lena_image_path)