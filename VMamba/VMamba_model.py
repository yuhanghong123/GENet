from Vmamba_modules import VSSM
import torch 

if __name__ == '__main__':
    # 创建一个VSSM对象
    vssm = VSSM()
    input = torch.randn(1, 3, 224, 224)
    output = vssm(input)
    print(output.shape)
