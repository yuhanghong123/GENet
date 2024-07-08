import cv2
import numpy as np

def add_noise_in_frequency_domain(image, a=0.5, b=0.8, beta_mean=0.1, beta_std=0.05):
    # 分离RGB通道
    r_channel, g_channel, b_channel = cv2.split(image)
    
    # 对每个通道进行傅里叶变换
    r_transform = np.fft.fft2(r_channel)
    r_transform_shift = np.fft.fftshift(r_transform)
    
    g_transform = np.fft.fft2(g_channel)
    g_transform_shift = np.fft.fftshift(g_transform)
    
    b_transform = np.fft.fft2(b_channel)
    b_transform_shift = np.fft.fftshift(b_transform)
    
    # 获取每个通道的幅度和相位
    r_amplitude = np.abs(r_transform_shift)
    r_phase = np.angle(r_transform_shift)
    
    g_amplitude = np.abs(g_transform_shift)
    g_phase = np.angle(g_transform_shift)
    
    b_amplitude = np.abs(b_transform_shift)
    b_phase = np.angle(b_transform_shift)
    
    alpha = np.random.uniform(a, b)
    # 从正态分布 N(beta_mean, beta_std^2) 中采样 beta
    beta_noise = np.random.normal(beta_mean, beta_std, size=r_amplitude.shape)
    
    # 添加噪声到幅度和相位
    r_amplitude_noise = alpha * r_amplitude + beta_noise 
    g_amplitude_noise = alpha * g_amplitude + beta_noise 
    b_amplitude_noise = alpha * b_amplitude + beta_noise 
    
    r_phase_noise = alpha * r_phase + beta_noise 
    g_phase_noise = alpha * g_phase + beta_noise
    b_phase_noise = alpha * b_phase + beta_noise
    
    # 重建每个通道的频域表示
    r_transform_noise = r_amplitude_noise * np.exp(1j * r_phase_noise)
    g_transform_noise = g_amplitude_noise * np.exp(1j * g_phase_noise)
    b_transform_noise = b_amplitude_noise * np.exp(1j * b_phase_noise)
    
    # 逆傅里叶变换得到增强后的图像
    r_transform_ishift = np.fft.ifftshift(r_transform_noise)
    r_img_back = np.fft.ifft2(r_transform_ishift)
    r_img_back = np.abs(r_img_back)
    
    g_transform_ishift = np.fft.ifftshift(g_transform_noise)
    g_img_back = np.fft.ifft2(g_transform_ishift)
    g_img_back = np.abs(g_img_back)
    
    b_transform_ishift = np.fft.ifftshift(b_transform_noise)
    b_img_back = np.fft.ifft2(b_transform_ishift)
    b_img_back = np.abs(b_img_back)
    
    # 合并RGB通道
    img_back = cv2.merge((r_img_back, g_img_back, b_img_back))
    img_back = img_back.astype(np.uint8)
    
    return img_back






# 读取图像
image = cv2.imread('/home/hyh1/gaze_domain_adption/code/PureGaze/output/epoch_1_iter_0.jpg')

# 对图像添加频域噪声
image_with_noise = add_noise_in_frequency_domain(image)

# 保存结果
cv2.imwrite('image_with_frequency_noise.jpg', image_with_noise)
