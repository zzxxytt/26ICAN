#ifndef INTERNAL_HPP_
#define INTERNAL_HPP_

#include "pid.h"

#define DEG_TO_RAD  0.0174532925f
#define RAD_TO_DEG  57.29577951f
#define PI 3.14159265358979323846f
#define MAX_IDX 2500
#define FILE_PATH "/home/orangepi/wheel_robot/pt.bin"//要改别忘了
#define DIS_PER_REC 4
#define BEZIER_PREVIEW 8
#define TUNNEL_LENGTH 15
#define TUNNEL_IDX 125
#define SSB_LENGTH 2500
#define SSB_IDX 2500
#define CROSSING_IDX 110
#define STOP_KP 0.05f


typedef struct 
{
    float x;
    float y;
} point_t;

typedef struct 
{
    int ssb_idx;
    int tunnel_idx;
} spe_ele_idx_t;


class Internal
{
private:
    point_t _pt_list[MAX_IDX]{};
    point_t _now_pos;
    point_t _target_pos;
    int _rec_idx = 0;
    int _dis_idx = 0;
    float _yaw_origin;
    float _dis_count=0;
    int _enable_rec=0;
    int _enable_dis=0;
    int _ssb_idx;
    int _tunnel_idx;
    int _crossing_idx;
    bool _stop_done;
    Pid pid_yaw;
    float convert2related_rad(float yaw);
    void add_point(point_t pt);
public:
    Internal(int ssb_idx, int crossing_idx, int tunnel_idx);
    int pt_count=0;
    void init(float yaw_origin);
    int get_rec_idx(void);
    void set_ssb(void);
    void set_tunnel_idx(void);
    void set_crossing_idx(void);
    bool is_stop(void);
    bool is_ssb(void);
    bool is_tunnel(void);
    void update_pos(float yaw,float Speed,float DT);
    void print_pos_now(void);
    bool save_pt_list(const char* file);
    int load_pt_list(const char* file);
    void enable_rec(void){_enable_rec=1;};
    void disable_rec(void){_enable_rec=0;};
    void enable_dis(void){_enable_dis=1;};
    void disable_dis(void){_enable_dis=0;};
    bool is_dis_enable(void){return _enable_dis;};
    void print_path(void);
    void set_pid_yaw(void);
    float get_pid_yaw_output(float target_yaw_rad, float measure_yaw_rad);
    void Bezier_change(void);
    void display_pos(void);
    bool idx_is_valid(int idx);
    float get_target_yaw();
    point_t get_pos_now(void);
    float get_dis_between_2pt(point_t pt1, point_t pt2);

};

#endif  // INTERNAL_HPP_
