import pandas as pd
import os
import shutil
import subprocess
from multiprocessing import Pool
from functools import partial

def convert_video_to_audio(id_num, video_dir, audio_dir):
    # 在视频目录中查找匹配的文件
    for filename in os.listdir(video_dir):
        parts = filename.split('_')
        if len(parts) >= 3 and parts[2] == str(id_num):
            video_path = os.path.join(video_dir, filename)
            wav_path = os.path.join(audio_dir, f"{id_num}.wav")
            
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-vn',
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                '-y',
                wav_path
            ]
            
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                print(f"成功转换 {filename} 到 {id_num}.wav")
                return True
            except subprocess.CalledProcessError as e:
                print(f"转换 {filename} 失败: {e}")
                return False
            
    return False

def main():
    # 读取CSV文件获取ID列表
    df = pd.read_csv('/home/user/xuxiao/DALL/dataset/CS-NRAC/scales.csv')
    ids = df['cust_id'].tolist()

    # 源视频目录和目标音频目录
    video_dir = '/mnt/audio-video/NJMU/2022年新生入学筛查/reading_vedios'
    audio_dir = '/home/user/xuxiao/DALL/dataset/CS-NRAC/audio'

    # 确保目标目录存在
    os.makedirs(audio_dir, exist_ok=True)

    # 创建进程池
    with Pool() as pool:
        # 使用偏函数固定video_dir和audio_dir参数
        convert_func = partial(convert_video_to_audio, video_dir=video_dir, audio_dir=audio_dir)
        # 并行执行转换任务
        results = pool.map(convert_func, ids)

if __name__ == '__main__':
    main()
