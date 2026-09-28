/* SO-101 control loop around a pocketpolicy-generated policy.
 *
 * Board-specific pieces are the hal_* functions, declared here and supplied
 * by the port (UART to the servo bus, a 30 Hz timer, the upstream detector
 * that reports object and goal positions). Nothing here allocates.
 *
 * Joint zero: the policy works in encoder ticks relative to the URDF zero.
 * CAL_ZERO holds each servo's raw count at that pose, from the arm's
 * calibration; the values below are placeholders (mid-range).
 */
#include <stdint.h>
#include "pocket.h"
#include "sts3215.h"

void hal_uart_write(const uint8_t *buf, int n);
int hal_uart_read(uint8_t *buf, int n, int timeout_ms);
void hal_wait_tick(void);                       /* blocks until the next 1/30 s */
int hal_detector(int16_t obj_mm[3], int16_t goal_mm[3]);  /* 0 when fresh */
int hal_grasped(void);

static const uint8_t IDS[6] = {1, 2, 3, 4, 5, 6};  /* LeRobot motor ids */
static const int16_t CAL_ZERO[6] = {2048, 2048, 2048, 2048, 2048, 2048};

static int16_t raw[POCKET_IN];
static int16_t plan[POCKET_OUT];
static uint8_t pkt[8 + 3 * 6];

static int read_joints(int16_t q[6])
{
    uint8_t rx[8];
    for (int j = 0; j < 6; ++j) {
        uint16_t p;
        hal_uart_write(pkt, sts_read_position(pkt, IDS[j]));
        if (hal_uart_read(rx, 8, 5) != 8 || sts_parse_position(rx, 8, IDS[j], &p) != 0)
            return -1;
        q[j] = (int16_t)((int16_t)p - CAL_ZERO[j]);
    }
    return 0;
}

static void write_targets(const int16_t *t)
{
    uint16_t pos[6];
    for (int j = 0; j < 6; ++j) {
        int32_t v = (int32_t)t[j] + CAL_ZERO[j];
        pos[j] = (uint16_t)(v < 0 ? 0 : (v > 4095 ? 4095 : v));
    }
    hal_uart_write(pkt, sts_sync_write_goal(pkt, IDS, pos, 6));
}

void so101_run(void)
{
    const int exec_k = POCKET_H > 1 ? POCKET_H / 2 : 1;
    for (;;) {
        if (read_joints(raw) != 0 || hal_detector(raw + 6, raw + 9) != 0) {
            hal_wait_tick();          /* hold position on a sensing fault */
            continue;
        }
        raw[12] = (int16_t)hal_grasped();
        pocket_step(raw, plan);
        for (int k = 0; k < exec_k; ++k) {
            write_targets(plan + 6 * k);
            hal_wait_tick();
        }
    }
}
