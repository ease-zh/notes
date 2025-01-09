调研最新基于大规模数据训练的TTS大模型。
## 1.  第一代TTS
第一代TTS主要基于人类先验知识设计专家系统，包括基于波形拼接方法和基于HMM参数统计方法。
## 2.  第二代TTS
第二代TTS主要基于深度学习网络，模型结构较小，一般专门录制的高质量数据进行训练，数据一般包含几个~几十个发音人。数据采集成本高、标注难度大。
第二代TTS的一般流程为：
![](https://gitee.com/ease_zh/image-host/raw/master/img/202412241341774.png)
其中主要聚焦与Acoustic Model，根据其结构可以分为自回归模型和非自回归模型。
### 2.1.  自回归模型
- 代表：Tacotron
- 方法：将上一帧预测结果作为当前帧预测的输入；额外预测stop token标志结束位。
- 优点：
	- 具有更好的韵律和自然度
	- 天然的支持流式合成，更容易集成
- 缺点：
	- 训练和推理都比较慢
	- 稳健性较差，如重复发音、漏字，结尾停止不恰当等
### 2.2.  非自回归模型：
- 代表：FastSpeech，NaturalSpeech
- 方法：显式预测每一个音素对应的时长，将每个音素的Linguistic Features根据时长对应扩展；扩展方式包括直接复制、Gauss混合等。
- 优点：
	- 解决了自回归模型合成不稳定的问题
	- 支持语速控制
	- 训练、推理可以并行，速度快
- 缺点：
	- 架构不原生支持流式合成，适配流式较复杂，且失去了并行推理速度上的优越性
	- 硬边界导致发音韵律自然度不如自回归模型
	- 依赖额外的对齐信息，标注工作量大、预处理较为复杂、推理时依赖单独的时长预测模块
### 2.3.  最新研究方向
普遍认为，基于个别人、小规模、高质量的语音数据训练的第二代TTS已经能够在自然度、可懂度上达到真人水平。
- 针对Acoustic Model的改进，包括引入扩散模型、流匹配模型等；
- 针对Acoustic Feature的改进，包括使用codec特征、使用vae特征等。

## 3.  第三代TTS
第三代TTS受大模型涌现能力的启发，使用多数人、大规模、低质量的语音数据训练大规模TTS模型，更多的聚焦在zero-shot TTS领域。
类似于第二代TTS的划分，第三代TTS也可以分为两大类：
- 基于Language Model（自回归结构）
- 基于Diffusion Model（非自回归结构
	这部分工作又可以进一步分为两类：
	- 依赖音素级时长信息，如NaturalSpeech2、3
	- 不依赖音素级时长信息，如SeedTTS-DiT、港中文蒙美玲团队的[[2024-SimpleSpeech-2--Towards-Simple-and-Efficient-Text-to-Speech-with-Flow-based-Scalar-Latent-Transformer-Diffusion-Mod-LFMGZL2J|SimpleSpeech]]系列、KRAFTON的[[DiTTo-TTS]]、微软的[[2024-E2-TTS--Embarrassingly-Easy-Fully-Non-Autoregressive-Zero-Shot-TTS-5293UPF5|E2 TTS]]、港中文武执政团队的[[2024-MaskGCT--Zero-Shot-Text-to-Speech-with-Masked-Generative-Codec-Transformer-HZBNLUU9|MaskGCT]]。
		- 基于句级时长进行时长控制，而不使用音素级时长，从而避免时长硬边界对自然度的影响。
- 两者相结合，如字节跳动的[[2024-Seed-TTS--A-Family-of-High-Quality-Versatile-Speech-Generation-Models-YQYSXQ3V|SeedTTS]]，阿里的[[2024-CosyVoice-2--Scalable-Streaming-Speech-Synthesis-with-Large-Language-Models-LNJ6Z7M8|CosyVoice2]]

工业界工作有ChatTTS。

第三代TTS的训练策略主要分为两种：
- Masked Generative Transformers（MaskGCT、E2TTS）
	- 以添加mask的方式对输入数据进行加噪，然后使用Transformer Backbone预测masked 部分
	- 推理时逐步解码，由纯空白输入先生成中间结果，再由中间结果生成最终结果
	- 与Flow-based和Diffusion-based思路一致
- Flow-based Transformers (SimpleSpeech，SeedTTS-DiT（疑似）)
	- 将语音信号与高斯噪声进行线性插值，根据时间步设置插值比例，从而得到随时间步逐渐噪化的一系列数据
	- 推理时候反向求解ODE，从噪声中逐步恢复语音信号
	- SeedTTS-DiT未明确说明，但大概率也是这种方式
- 
Mask&Predict、
SoundStorm、NaturalSpeech1、2、3，AudioLDM，VaLLE、XTTS、Encodec、Vocos