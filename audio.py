import torchaudio
import gradio as gr
from audiocraft.models import AudioGen
import torch
import numpy as np
import os
from tempfile import NamedTemporaryFile

# 初始化模型
model = AudioGen.get_pretrained('facebook/audiogen-medium')

def generate_audio(description, duration):
    # 设置生成参数
    model.set_generation_params(duration=duration)
    
    # 生成音频
    wav = model.generate([description])
    # 获取音频数据，shape应该是 [channels, samples]，并移到CPU
    audio = wav[0].squeeze(0).cpu()  # 移除批次维度并转到CPU
    
    # 确保采样率在合理范围内
    if model.sample_rate > 48000:
        target_sr = 48000
        resampler = torchaudio.transforms.Resample(
            orig_freq=model.sample_rate,
            new_freq=target_sr
        )
        audio = resampler(audio)
        sample_rate = target_sr
    else:
        sample_rate = model.sample_rate
    
    # 创建临时文件
    with NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        # 保存音频
        torchaudio.save(temp_file.name, audio.unsqueeze(0), sample_rate)
        return temp_file.name

# 创建 Gradio 界面
demo = gr.Interface(
    fn=generate_audio,
    inputs=[
        gr.Textbox(label="音频描述（英文）", placeholder="例如：dog barking"),
        gr.Slider(minimum=1, maximum=10, value=5, step=1, label="音频时长（秒）")
    ],
    outputs=gr.Audio(label="生成的音频"),
    title="AudioGen 音频生成器",
    description="输入文本描述（英文）来生成相应的音频。",
    examples=[
        ["dog barking", 5],
        ["sirene of an emergency vehicle", 5],
        ["footsteps in a corridor", 5]
    ]
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7160)
    
    # 清理临时文件
    for file in os.listdir('/tmp'):
        if file.endswith('.wav'):
            try:
                os.remove(os.path.join('/tmp', file))
            except:
                pass