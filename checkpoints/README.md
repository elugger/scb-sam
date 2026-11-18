# 初始化权重与代码分离

代码仓库不包含模型二进制。此前为整理核验下载的7份权重现仅保存在仓库外的本地资产目录，尚未上传或公开发布。manifest.json记录来源、真实原始SHA和分类。

## 三类初始化

1. 随机初始化：使用代码中的初始化规则和固定种子，不需要下载初始随机参数文件。不同CUDA/硬件并不因此保证逐比特相同。
2. 上游预训练：SAM ViT-B（SFR）、原版SAM-Med3D（ReLiF-3D）、SAM-Med3D Turbo（我们的方法和对应Turbo消费者）。复现相应方法需要相应权重，但代码库只需给出精确版本、官方获取途径、SHA及放置位置，不需要再分发上游二进制。三者不能混换。
3. 本项目监督阶段产物：BY5、BY10、Public5、Public10四个监督初始化。若提供完整监督阶段训练入口、数据划分、参数和接续规则，读者可自行生成；下载可以作为节省算力和复用历史同一起点的便利项。现有历史引擎直接读取固定SHA文件，新机监督阶段的完整可移植重建尚未验收，不能直接删掉依赖后声称已支持从头训练。

H/HP/HQ/HPQ应复用同一次BY5监督阶段产物，避免各自生成不同起点。提供同一文件是最方便的历史起点复用方式，但也可由完整脚本先生成一次后四组共同加载。

最终模型权重与初始化不是同一件事。最终模型下载可方便直接推理/核表，并非每个对照或开发失败版本都需要发布。HQ0.25历史未保存末轮权重，只能给出真实训练配置与结果，不能补造下载。

## 获取与验证工具

在项目根目录运行：

```bash
python scripts/checkpoint_assets.py list
python scripts/checkpoint_assets.py verify --name sam_med3d_turbo.pth --path /outside/repository/weights/sam_med3d_turbo.pth
python scripts/checkpoint_assets.py download --name sam_med3d_turbo.pth --directory /outside/repository/weights
```

只有manifest登记了官方URL的条目才可下载。每次下载以历史字节数和SHA256验收，不因当前上游文件同名就接受替换。工具不会加载pickle，不覆盖已有不匹配文件，不把权重写入代码仓库。联网传输可用性与本地文件校验分别记录。原版SAM-Med3D的匹配下载版本尚未确认时，应提供已有文件进行verify，不能改用Turbo。

源码Git仓库保留脚本、配置、模型源码、依赖说明、数据准备与评价程序，以及下载/校验说明。大权重通过上游官方链接或独立release/model repository提供，受对应许可约束；本项目自有Apache-2.0不自动覆盖第三方权重。

参考：[SAM官方代码与权重分离](https://github.com/facebookresearch/segment-anything)、[SAM-Med3D官方checkpoint说明](https://github.com/uni-medical/SAM-Med3D)、[GitHub大文件说明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)。
