#include "internal.hpp"
#include <cmath>
#include <iostream>
#include <fstream>

float normalize_rad(float yaw)
{
    while (yaw > PI)
    {
        yaw -= 2.0f * PI;
    }
    while (yaw < -PI)
    {
        yaw += 2.0f * PI;
    }
    return yaw;
}

Internal::Internal(int ssb_idx, int crossing_idx, int tunnel_idx, int stair_idx)
{
    _ssb_idx = ssb_idx;
    _crossing_idx = crossing_idx;
    _tunnel_idx = tunnel_idx;
    _stair_idx = stair_idx;
    _now_pos.x = 0;
    _now_pos.y = 0;
    _yaw_origin = 0;
    pt_count = 0;
    _stop_done = false;
    set_pid_yaw();
    pt_count = load_pt_list(FILE_PATH);

    std::cout << "Loaded " << pt_count << " points from file: " << FILE_PATH << "\n";
    // for(int i=0;i<pt_count;i++)
    // {
    //     std::cout << "x:" << _pt_list[i].x << " y:" << _pt_list[i].y << "\n";
    // }
}

void Internal::init(float yaw_origin)
{
    _now_pos.x = 0; 
    _now_pos.y = 0;
    _yaw_origin = yaw_origin;
    _rec_idx = 0;
    _dis_idx = 0;
    _dis_count = 0;
    _stop_done = false;
}

bool Internal::save_pt_list(const char *file)
{
    int pt_count = _rec_idx; // 使用当前索引作为点数
    this->pt_count = pt_count; // 更新类成员变量
    if (pt_count < 0 || pt_count > MAX_IDX)
        return false;

    std::ofstream ofs(file, std::ios::binary | std::ios::trunc);
    if (!ofs)
        return false;

    ofs.write((const char *)&pt_count, sizeof(pt_count));          // 先存点数(4字节)
    ofs.write((const char *)_pt_list, pt_count * sizeof(point_t)); // 再存坐标
    return (bool)ofs;                                              // 离开作用域自动 flush + close
}

// ---------- 恢复：返回读到的点数，失败返回 -1 ----------
int Internal::load_pt_list(const char *file) // load_pt_list(FILE_PATH)
{
    std::ifstream ifs(file, std::ios::binary);
    if (!ifs)
        return -1;

    int n = 0;
    if (!ifs.read((char *)&n, sizeof(n)))
        return -1;
    if (n < 0 || n > MAX_IDX)
        return -1; // 防脏数据越界
    if (!ifs.read((char *)_pt_list, n * sizeof(point_t)))
        return -1;

    pt_count = n;
    return n;
}

int Internal::get_rec_idx(void)
{
    return _rec_idx;
}

void Internal::add_point(point_t pt)
{
    if (_rec_idx >= MAX_IDX)
        return;
    _pt_list[_rec_idx].x = pt.x;
    _pt_list[_rec_idx].y = pt.y;
    _rec_idx++;
}

void Internal::set_ssb(void)
{
    _ssb_idx = _rec_idx;
}

void Internal::set_tunnel_idx(void)
{
    _tunnel_idx = _rec_idx;
}

void Internal::set_crossing_idx(void)
{
    _crossing_idx = _rec_idx;
}

void Internal::set_stair_idx(void)
{
    _stair_idx = _rec_idx;
}

float Internal::convert2related_rad(float yaw)
{
    float yaw_related = yaw - _yaw_origin;
    yaw_related = normalize_rad(yaw_related);
    return yaw_related;
}

void Internal::update_pos(float yaw_rad, float Speed, float DT) // 机体的弧度制航向角 机体速度 时间间隔
{
    float related_yaw = convert2related_rad(yaw_rad);
    _now_pos.y += Speed * std::cos(related_yaw) * DT;
    _now_pos.x += Speed * std::sin(related_yaw) * DT;
    _dis_count += Speed * DT;
    if (_dis_count >= DIS_PER_REC)
    {
        if (_enable_rec)
        {
            add_point(_now_pos);
            _dis_count -= DIS_PER_REC;
        }
    }
}

void Internal::display_pos(void)//装载目标点
{
    double best_dis = 1e9;
    int best_idx = _dis_idx;
    bool valid_idx_found = idx_is_valid(_dis_idx);
    if (!valid_idx_found)
    {
        disable_dis();
        return;
    } 
    for (int i = _dis_idx; i < _dis_idx + 2; i++)
    {
        double dis = std::sqrt(std::pow(_now_pos.x - _pt_list[i].x, 2) + std::pow(_now_pos.y - _pt_list[i].y, 2));
        if (dis < best_dis)
        {
            best_dis = dis;
            best_idx = i;
        }
    }
    _dis_idx = best_idx;
    Bezier_change();
}

void Internal::set_pid_yaw(void)
{
    pid_yaw.kp = 3.0f;
    pid_yaw.ki = 0.0f;
    pid_yaw.kd = 0.0f;
}

float Internal::get_pid_yaw_output(float target_yaw_rad, float measure_yaw_rad) // 机体的弧度制航向角
{
    float related_measure_yaw = convert2related_rad(measure_yaw_rad);
    return pid_yaw.update(target_yaw_rad, related_measure_yaw);
}

bool Internal::idx_is_valid(int idx)
{
    return (idx >= 0 && idx < (pt_count - 2 * BEZIER_PREVIEW));
}

void Internal::Bezier_change(void)
{
    float BEZIER_t = 0.5f;

    float x1 = _now_pos.x;
    float y1 = _now_pos.y;
    float x2 = _pt_list[_dis_idx + BEZIER_PREVIEW].x;
    float y2 = _pt_list[_dis_idx + BEZIER_PREVIEW].y;
    float x3 = _pt_list[_dis_idx + 2 * BEZIER_PREVIEW].x;
    float y3 = _pt_list[_dis_idx + 2 * BEZIER_PREVIEW].y;

    _target_pos.x = (1 - BEZIER_t) * (1 - BEZIER_t) * x1 + 2 * (1 - BEZIER_t) * BEZIER_t * x2 + BEZIER_t * BEZIER_t * x3;
    _target_pos.y = (1 - BEZIER_t) * (1 - BEZIER_t) * y1 + 2 * (1 - BEZIER_t) * BEZIER_t * y2 + BEZIER_t * BEZIER_t * y3;
}

float Internal::get_target_yaw()
{
    float delta_x = _target_pos.x - _now_pos.x;
    float delta_y = _target_pos.y - _now_pos.y;
    float target_yaw_rad = std::atan2(delta_x, delta_y);
    return target_yaw_rad;
}

bool Internal::is_tunnel(void)
{
    if (_dis_idx >= _tunnel_idx && _dis_idx < _tunnel_idx + TUNNEL_LENGTH)
    {
        return true;
    }
    else
    {
        return false;
    }
}

bool Internal::is_stair(void)
{
    if (_dis_idx >= _stair_idx && _dis_idx < _stair_idx + STAIR_LENGTH)
    {
        return true;
    }
    else
    {
        return false;
    }    

}

bool Internal::is_stop(void)
{
    if(_dis_idx >= _crossing_idx && (!_stop_done))
    {
        _stop_done = true;
        return true;
    }
    else
    {
        return false;
    }
}

void Internal::print_pos_now(void)
{
    std::cout << "x:" << _now_pos.x << " y:" << _now_pos.y << "\n";
}

void Internal::print_path(void)
{
    for (int i = 0; i < _rec_idx; i++)
    {
        std::cout << "x:" << _pt_list[i].x << " y:" << _pt_list[i].y << "\n";
    }
}
