# 一帧图像的完整 YOLO 识别结果。
# 一个话题一次发布即可拿到「本帧全部目标」，控制节点无需自己攒列表。

std_msgs/Header header

# 推理来源：camera 表示本节点直接采图，image_topic 表示订阅外部图像话题
string source

# 本帧图像尺寸（像素），检测框即基于该尺寸
uint32 image_width
uint32 image_height

# 本次推理耗时，单位毫秒（预处理 + NPU 前向 + 后处理）
float32 inference_ms

# 实际参与推理的输入边长，例如 640
uint32 input_size

# 使用的后端：npu / onnx / mock
string backend

Detection[] detections
