// 识别结果接收端：订阅 recognize_yolo 的检测结果，缓存最新一帧，判定新鲜度并打印。
//
// ============================================================================
// 职责边界（这是本节点最重要的设计约束）
// ============================================================================
//
//   **本节点只接收，至始至终不做控制。**
//
//   它不发布 /cmd_vel，不发布任何关节或姿态指令，也不参与决策。
//   它的全部价值是把 Python 侧 10 Hz、随时可能断流的感知数据，变成一个
//   高频消费者可以安全查询的"最新一帧"。
//
//   刻意与 wl_base_node 分离：那个节点是实时、安全关键的（串口通下位机、
//   控电机、读 IMU、发 /odom）。感知是非实时、可能失效的。混在一起会让
//   视觉的抖动污染控制周期，视觉崩了还会拖垮底盘 —— 感知失效不该影响
//   底盘能不能站住。
//
// ============================================================================
// 订阅 / 参数
// ============================================================================
//
//   订阅： input_topic（默认 /recognize_yolo/detections）
//          recognize_yolo_msgs/msg/DetectionArray
//
//   参数：
//     input_topic   订阅的话题名。
//     target_class  关心的类别名。**留空表示不筛选**，本帧全部目标都会打印。
//                   注意：mock 后端给的是 class_0 / class_1；加载真模型并
//                   指定 class_names_file 之后会变成 person / bottle 这类
//                   真实名字。所以不要把它硬编码进代码，用参数传。
//     min_score     低于该分数的目标不展示。0 表示不按分数过滤。
//     max_age       超过该秒数没收到新帧，判定为断流。
//     print_period  打印周期（秒）。0 表示每帧都打印。
//     health_period 健康检查周期（秒）。
//
// ============================================================================
// 三个必须记住的实现要点
// ============================================================================
//
//   1) 回调只存不算。回调频率跟着推理走（10 Hz），消费者频率远高于它。
//      在回调里做解析或计算，会让消费侧被 10 Hz 绑架。
//
//   2) 新鲜度用 std::chrono::steady_clock 的本地接收时间判断，**不用**
//      header.stamp。后者来自图像时间戳，跨机器或时钟不同步时会骗你。
//
//   3) 「没检测到目标」和「视觉断流」是两件完全不同的事：
//        收到帧但 detections 为空 -> 画面里真的没有东西，可以继续搜索
//        很久没收到帧            -> 视觉节点挂了，必须停下
//      两者的安全动作不同，所以状态必须分开表达（见 VisionState）。
//
// ============================================================================

#include <chrono>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include <recognize_yolo_msgs/msg/detection.hpp>
#include <recognize_yolo_msgs/msg/detection_array.hpp>

using recognize_yolo_msgs::msg::Detection;
using recognize_yolo_msgs::msg::DetectionArray;

namespace
{

/// 固定小数位格式化，避免把 float32 打印成 0.75999999 这类噪声。
std::string num(double value, int precision)
{
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(precision) << value;
  return stream.str();
}

}  // namespace

/// 感知状态。把「没目标」和「断流」分开，是本节点的关键设计。
enum class VisionState
{
  kNoData,      ///< 从未收到任何一帧：对面还没起来
  kStale,       ///< 收到过，但已超过 max_age：视觉节点疑似挂了
  kEmpty,       ///< 帧是新鲜的，但画面里确实没有目标
  kHasTargets,  ///< 帧是新鲜的，且有目标
};

class VisionClientNode : public rclcpp::Node
{
public:
  VisionClientNode()
  : Node("vision_client")
  {
    // ------------------------------------------------------------------
    // 参数
    // ------------------------------------------------------------------
    input_topic_ = this->declare_parameter<std::string>(
      "input_topic", "/recognize_yolo/detections");
    target_class_ = this->declare_parameter<std::string>("target_class", "");
    min_score_ = this->declare_parameter<double>("min_score", 0.0);
    max_age_ = this->declare_parameter<double>("max_age", 0.5);
    print_period_ = this->declare_parameter<double>("print_period", 0.0);
    health_period_ = this->declare_parameter<double>("health_period", 0.5);

    // 防御：这两个值来自参数，配错会让节点行为失常而不是报错。
    //   health_period <= 0 -> 定时器变成忙循环，在板子上会吃满一个核
    //   max_age      <= 0  -> 所有帧都被判成过期，永远报断流
    if (health_period_ < 0.05) {
      RCLCPP_WARN(
        get_logger(), "health_period=%s 过小，已钳到 0.05 秒",
        num(health_period_, 3).c_str());
      health_period_ = 0.05;
    }
    if (max_age_ <= 0.0) {
      RCLCPP_WARN(get_logger(), "max_age 必须为正数，已回落到 0.5 秒");
      max_age_ = 0.5;
    }

    // ------------------------------------------------------------------
    // 订阅
    //
    // QoS 必须与 Python 发布端一致。Python 侧是 create_publisher(..., 10)，
    // 也就是默认的 reliable + volatile + depth 10；这里的 rclcpp::QoS(10)
    // 同样是 reliable + volatile + depth 10，所以能直接通。
    //
    // 如果以后某一侧改成 best_effort 或 sensor-data QoS，另一侧必须跟着改，
    // 否则表现为「ros2 topic list 里有它、ros2 topic hz 也有，但这里就是
    // 收不到」—— 这是 ROS 2 里最经典的隐形坑。
    // ------------------------------------------------------------------
    subscription_ = this->create_subscription<DetectionArray>(
      input_topic_,
      rclcpp::QoS(10),
      std::bind(&VisionClientNode::on_detections, this, std::placeholders::_1));

    // ------------------------------------------------------------------
    // 健康检查定时器
    //
    // 用一个独立定时器判定断流，而不是让消费者自己去算，
    // 这样「视觉挂了」会主动出现在日志里，不用等到出事故才发现。
    // ------------------------------------------------------------------
    health_timer_ = this->create_wall_timer(
      std::chrono::duration<double>(health_period_),
      std::bind(&VisionClientNode::on_health_timer, this));

    const std::string class_note =
      target_class_.empty() ? std::string("（空，本帧全部目标都打印）") : target_class_;
    const std::string score_note =
      min_score_ > 0.0 ? num(min_score_, 2) : std::string("（不按分数过滤）");
    const std::string print_note =
      print_period_ > 0.0 ? num(print_period_, 2) : std::string("（每帧都打印）");

    RCLCPP_INFO(
      get_logger(),
      "识别结果接收端已启动 —— 只接收，不做控制\n"
      "  订阅话题  : %s\n"
      "  类别筛选  : %s\n"
      "  最低分数  : %s\n"
      "  断流阈值  : %s 秒\n"
      "  打印周期  : %s",
      input_topic_.c_str(), class_note.c_str(), score_note.c_str(),
      num(max_age_, 2).c_str(), print_note.c_str());
  }

  // ------------------------------------------------------------------
  // 供后续消费者使用的查询接口
  //
  // 当前节点自身不做控制，这两个接口是为「以后要接的东西」准备的。
  // 注意它们返回的是**整帧**，不是单个目标 —— 一帧里出现多个物品是常态，
  // 由调用方自己决定怎么用，这里不做「只挑一个」的假设。
  // ------------------------------------------------------------------

  /// 线程安全地取最新一帧。
  ///
  /// @param max_age_sec 新鲜度阈值
  /// @param fresh       回填该帧是否还在阈值内；为 false 说明拿到的是过期旧帧
  /// @return 最新一帧；空指针表示从未收到过任何一帧。
  ///         注意返回空**不等于**「画面里没目标」，后者是 detections 为空。
  DetectionArray::ConstSharedPtr latest_frame(
    double max_age_sec, bool * fresh = nullptr) const
  {
    std::lock_guard<std::mutex> lock(mutex_);

    if (fresh != nullptr) {
      *fresh = has_latest_ && (age_locked() <= max_age_sec);
    }
    return latest_;
  }

  /// 当前感知状态。四种取值的区别见 VisionState 的注释。
  VisionState state(double max_age_sec) const
  {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!has_latest_) {
      return VisionState::kNoData;
    }
    if (age_locked() > max_age_sec) {
      return VisionState::kStale;
    }
    return latest_->detections.empty() ? VisionState::kEmpty : VisionState::kHasTargets;
  }

private:
  // ------------------------------------------------------------------
  // 订阅回调：只做一件事 —— 把最新一帧存下来
  // ------------------------------------------------------------------
  void on_detections(DetectionArray::ConstSharedPtr message)
  {
    bool print_now = true;

    {
      std::lock_guard<std::mutex> lock(mutex_);

      // 只换 shared_ptr 的指向，不拆字段、不原地修改消息内容。
      // 拆字段会白白丢掉 ROS 2 的零拷贝优势，还容易在消息加字段时漏掉；
      // 原地改会和正在读缓存的消费者产生数据竞争。
      latest_ = message;
      latest_at_ = std::chrono::steady_clock::now();
      has_latest_ = true;
      ++received_;

      if (print_period_ > 0.0) {
        const auto now = std::chrono::steady_clock::now();
        print_now = (now - last_print_at_) >= std::chrono::duration<double>(print_period_);
        if (print_now) {
          last_print_at_ = now;
        }
      }
    }

    // 打印放在锁外面，避免慢速控制台输出拖住其他回调。
    if (print_now) {
      print_frame(*message);
    }
  }

  /// 按参数筛选出要展示的目标。
  ///
  /// 用的是 class_name 而不是 class_id —— id 是模型内部索引，换个模型就全变了；
  /// name 才是语义，跨模型稳定。
  std::vector<const Detection *> select(const DetectionArray & message) const
  {
    std::vector<const Detection *> result;
    result.reserve(message.detections.size());

    for (const Detection & item : message.detections) {
      if (!target_class_.empty() && item.class_name != target_class_) {
        continue;
      }
      if (item.score < min_score_) {
        continue;
      }
      result.push_back(&item);
    }
    return result;
  }

  /// 打印一帧的全部目标。一帧里出现多个物品是常态，这里逐个列出而不做取舍。
  void print_frame(const DetectionArray & message)
  {
    const std::vector<const Detection *> selected = select(message);
    const bool filtering = !target_class_.empty() || min_score_ > 0.0;

    std::ostringstream out;
    out << "[第 " << received_ << " 帧]"
        << " backend=" << (message.backend.empty() ? "?" : message.backend)
        << " source=" << (message.source.empty() ? "?" : message.source)
        << " 图像=" << message.image_width << "x" << message.image_height
        << " 推理=" << num(message.inference_ms, 1) << "ms"
        << " input=" << message.input_size
        << " 目标总数=" << message.detections.size();

    if (filtering) {
      out << " 通过筛选=" << selected.size();
    }

    for (std::size_t index = 0; index < selected.size(); ++index) {
      const Detection & item = *selected[index];
      out << "\n    #" << index
          << "  " << (item.class_name.empty()
                        ? ("class_" + std::to_string(item.class_id))
                        : item.class_name)
          << "  score=" << num(item.score, 2)
          << "  中心=(" << num(item.center_x, 1) << ", " << num(item.center_y, 1) << ")"
          << "  归一=(" << num(item.norm_center_x, 3) << ", " << num(item.norm_center_y, 3) << ")"
          << "  框=" << num(item.xmax - item.xmin, 0) << "x"
          << num(item.ymax - item.ymin, 0)
          << "  track_id=" << item.track_id;
    }

    if (selected.empty()) {
      // 这两种"空"含义不同，日志里必须区分开 —— 排查时能省很多时间。
      out << (message.detections.empty()
                ? "\n    （本帧画面里没有检测到任何目标）"
                : "\n    （本帧有目标，但都不满足筛选条件）");
    }

    RCLCPP_INFO(get_logger(), "%s", out.str().c_str());
  }

  // ------------------------------------------------------------------
  // 健康检查：把「断流」主动暴露出来
  // ------------------------------------------------------------------
  void on_health_timer()
  {
    switch (state(max_age_)) {
      case VisionState::kNoData:
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "尚未收到任何检测结果。%s 起了吗？（用 ros2 topic list 确认话题存在）",
          input_topic_.c_str());
        break;

      case VisionState::kStale:
        // 断流：视觉节点疑似挂了。这跟「画面里没有目标」是完全不同的故障，
        // 消费侧必须区别对待 —— 前者要停，后者可以继续搜索。
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 1000,
          "检测结果已断流 %.2f 秒（阈值 %.2f 秒），累计收到 %llu 帧",
          age_seconds(), max_age_,
          static_cast<unsigned long long>(received_count()));
        break;

      case VisionState::kEmpty:
      case VisionState::kHasTargets:
        // 一切正常，不刷日志。帧内容由 print_frame 负责展示。
        break;
    }
  }

  // ------------------------------------------------------------------
  // 必须在持有 mutex_ 的前提下调用
  // ------------------------------------------------------------------
  double age_locked() const
  {
    return std::chrono::duration<double>(
      std::chrono::steady_clock::now() - latest_at_).count();
  }

  double age_seconds() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return has_latest_ ? age_locked() : std::numeric_limits<double>::infinity();
  }

  std::uint64_t received_count() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return received_;
  }

  // ------------------------------------------------------------------
  // 参数
  // ------------------------------------------------------------------
  std::string input_topic_;
  std::string target_class_;
  double min_score_{0.0};
  double max_age_{0.5};
  double print_period_{0.0};
  double health_period_{0.5};

  rclcpp::Subscription<DetectionArray>::SharedPtr subscription_;
  rclcpp::TimerBase::SharedPtr health_timer_;

  // ------------------------------------------------------------------
  // 缓存与统计。下面全部字段由 mutex_ 保护。
  //
  // 缓存的是**整帧**而不是挑出来的单个目标 —— 一帧里出现多个物品是常态，
  // 挑目标的活儿留给调用方，缓存层不做这个假设。
  // ------------------------------------------------------------------
  mutable std::mutex mutex_;
  DetectionArray::ConstSharedPtr latest_;
  std::chrono::steady_clock::time_point latest_at_{};
  std::chrono::steady_clock::time_point last_print_at_{};
  bool has_latest_{false};
  std::uint64_t received_{0};
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  // 单线程 executor 足够：回调里只存一个 shared_ptr，是微秒级的操作。
  // 这正是「回调只存不算」这条约束换来的好处 —— 不需要多线程，
  // 也就不需要操心并发。
  rclcpp::spin(std::make_shared<VisionClientNode>());

  rclcpp::shutdown();
  return 0;
}
