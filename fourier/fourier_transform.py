import numpy as np
import cv2

def fourier_transform(image):
    """对图像进行傅里叶变换"""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f_transform = np.fft.fft2(gray_image)
    f_transform_shift = np.fft.fftshift(f_transform)
    return f_transform_shift

def inverse_fourier_transform(f_transform_shift):
    """对频域图像进行逆傅里叶变换"""
    f_ishift = np.fft.ifftshift(f_transform_shift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return img_back

def low_high_pass_filter(f_transform_shift, radius):
    """低通和高通滤波器"""
    rows, cols = f_transform_shift.shape
    crow, ccol = rows // 2 , cols // 2  # 中心位置

    # 生成掩模
    mask = np.zeros((rows, cols), np.uint8)
    mask[crow-radius:crow+radius, ccol-radius:ccol+radius] = 1

    # 低频部分
    low_freq = f_transform_shift * mask
    # 高频部分
    high_freq = f_transform_shift * (1 - mask)

    return low_freq, high_freq

def apply_filters(image, radius):
    """应用滤波器并返回高低频图像"""
    f_transform_shift = fourier_transform(image)
    low_freq, high_freq = low_high_pass_filter(f_transform_shift, radius)

    # 逆变换得到空间域图像
    low_freq_image = inverse_fourier_transform(low_freq)
    low_freq_image = np.stack([low_freq_image] * 3, axis=-1)

    high_freq_image = image - low_freq_image

    return low_freq_image, high_freq_image

# 读取图像
image = cv2.imread('/home/hyh1/gaze_domain_adption/code/PureGaze/output/epoch_1_iter_1.jpg')
image = cv2.resize(image, (224, 224))  # 确保图像大小一致

# 应用高低频滤波器
radius = 30  # 设置频率阈值
low_freq_image, high_freq_image = apply_filters(image, radius)

# 将结果保存为文件
cv2.imwrite('low_frequency_image.jpg', low_freq_image)
cv2.imwrite('high_frequency_image.jpg', high_freq_image)
