import cv2
import gradio as gr
import os
from tqdm import tqdm
from modules import script_callbacks


def add_tab():
    with gr.Blocks(analytics_enabled=False) as ui:
        with gr.Row().style(equal_height=False):
            with gr.Column(variant='panel'):
                gr.HTML(value="<p>Extract frames from video</p>")
                input_video = gr.Textbox(label="Input video path")
                output_folder = gr.Textbox(label="Output frames folder path")
                output_fps = gr.Slider(minimum=1, maximum=120, step=1, label="Save every N frame", value=1)
                custom_resolution_checkbox = gr.Checkbox(label="Use custom resolution")
                custom_width_slider = gr.Slider(minimum=1, maximum=1920, step=1, label="Custom Width", value=512)
                custom_height_slider = gr.Slider(minimum=1, maximum=1080, step=1, label="Custom Height", value=768)
                extract_frames_btn = gr.Button(label="Extract Frames", variant='primary')

            with gr.Column(variant='panel'):
                gr.HTML(value="<p>Merge frames to video</p>")
                input_folder = gr.Textbox(label="Input frames folder path")
                output_video = gr.Textbox(label="Output video path")
                output_video_fps = gr.Slider(minimum=1, maximum=60, step=1, label="Video FPS", value=30)
                merge_frames_btn = gr.Button(label="Merge Frames", variant='primary')

            extract_frames_btn.click(
                fn=extract_frames,
                inputs=[
                    input_video,
                    output_folder,
                    output_fps,
                    custom_resolution_checkbox,
                    custom_width_slider,
                    custom_height_slider
                ]
            )

            merge_frames_btn.click(
                fn=merge_frames,
                inputs=[
                    input_folder,
                    output_video,
                    output_video_fps
                ]
            )

    return [(ui, "Video<->Frame", "vf_converter")]


def extract_frames(video_path: str, output_path: str, custom_fps=None,use_custom_resolution=False, custom_width=512, custom_height=768):
    """从视频文件中提取帧并输出为png格式

    Args:
        video_path (str): 视频文件的路径
        output_path (str): 输出帧的路径
        custom_fps (int, optional): 自定义输出帧率（默认为None，表示与视频帧率相同）
        use_custom_resolution (bool): 是否使用自定义分辨率
        custom_width (int): 自定义宽度
        custom_height (int): 自定义高度

    Returns:
        None
    """
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    custom_fps = int(custom_fps)
    frame_count = 0

    print(f"Extracting {video_path} to {output_path}...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if use_custom_resolution:
            frame = cv2.resize(frame, (custom_width, custom_height))

        if custom_fps:
            if frame_count % custom_fps == 0:
                output_name = os.path.join(output_path, '{:06d}.png'.format(frame_count))
                cv2.imwrite(output_name, frame)
        else:
            output_name = os.path.join(output_path, '{:06d}.png'.format(frame_count))
            cv2.imwrite(output_name, frame)
        frame_count += 1

    cap.release()
    print("Extract finished.")


def merge_frames(frames_path: str, output_path: str, fps=None):
    """将指定文件夹内的所有png图片按顺序合并为一个mp4视频文件

    Args:
        frames_path (str): 输入帧的路径（所有帧必须为png格式）
        output_path (str): 输出视频的路径（需以.mp4为文件扩展名）
        fps (int, optional): 输出视频的帧率（默认为None，表示与输入帧相同）

    Returns:
        None
    """

    # 获取所有png图片
    frames = [f for f in os.listdir(frames_path) if f.endswith('.png')]
    img = cv2.imread(os.path.join(frames_path, frames[0]))
    height, width, _ = img.shape
    fps = fps or int(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Merging {len(frames)} frames to video...")
    for f in tqdm(frames):
        img = cv2.imread(os.path.join(frames_path, f))
        video_writer.write(img)

    video_writer.release()
    print("Merge finished")


script_callbacks.on_ui_tabs(add_tab)