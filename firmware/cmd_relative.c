#include "cmd_relative.h"

#define N (CMD_LAT + 1)

static int16_t sat16(int32_t v)
{
    return (int16_t)(v > 32767 ? 32767 : (v < -32768 ? -32768 : v));
}

static void push(cmd_ring *r, const int16_t cmd[6])
{
    for (int j = 0; j < 6; ++j) r->buf[r->head][j] = cmd[j];
    r->head = (r->head + 1) % N;
}

void cmd_ring_init(cmd_ring *r, const int16_t q[6])
{
    r->head = 0;
    for (int i = 0; i < N; ++i) push(r, q);
}

void cmd_correction(const cmd_ring *r, const int16_t q_read[6], int16_t corr[6])
{
    for (int j = 0; j < 6; ++j) {
        int32_t c = (int32_t)r->buf[r->head][j] - q_read[j];
        corr[j] = (int16_t)(c > CMD_MAX_CORR ? CMD_MAX_CORR : (c < -CMD_MAX_CORR ? -CMD_MAX_CORR : c));
    }
    corr[CMD_GRIPPER] = 0;
}

void cmd_apply(cmd_ring *r, const int16_t target[6], const int16_t corr[6], int16_t cmd[6])
{
    for (int j = 0; j < 6; ++j) cmd[j] = sat16((int32_t)target[j] + corr[j]);
    push(r, cmd);
}

void cmd_hold(cmd_ring *r, int16_t cmd[6])
{
    const int16_t *last = r->buf[(r->head + N - 1) % N];
    for (int j = 0; j < 6; ++j) cmd[j] = last[j];
    push(r, cmd);
}
