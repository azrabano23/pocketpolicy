#include "sts3215.h"

static uint8_t checksum(const uint8_t *p, int n)
{
    unsigned s = 0;
    for (int i = 0; i < n; ++i) s += p[i];
    return (uint8_t)(~s & 0xFF);
}

int sts_sync_write_goal(uint8_t *buf, const uint8_t *ids, const uint16_t *pos, int n)
{
    int k = 0;
    buf[k++] = 0xFF;
    buf[k++] = 0xFF;
    buf[k++] = STS_BROADCAST;
    buf[k++] = (uint8_t)(3 * n + 4);  /* params: addr, width, n*(id, lo, hi) */
    buf[k++] = STS_INST_SYNC_WRITE;
    buf[k++] = STS_GOAL_POSITION;
    buf[k++] = 2;
    for (int i = 0; i < n; ++i) {
        buf[k++] = ids[i];
        buf[k++] = (uint8_t)(pos[i] & 0xFF);
        buf[k++] = (uint8_t)(pos[i] >> 8);
    }
    buf[k] = checksum(buf + 2, k - 2);
    return k + 1;
}

int sts_read_position(uint8_t *buf, uint8_t id)
{
    buf[0] = 0xFF;
    buf[1] = 0xFF;
    buf[2] = id;
    buf[3] = 4;
    buf[4] = STS_INST_READ;
    buf[5] = STS_PRESENT_POSITION;
    buf[6] = 2;
    buf[7] = checksum(buf + 2, 5);
    return 8;
}

int sts_parse_position(const uint8_t *buf, int len, uint8_t id, uint16_t *pos)
{
    /* FF FF ID LEN=4 ERR lo hi CHK */
    if (len < 8 || buf[0] != 0xFF || buf[1] != 0xFF || buf[2] != id || buf[3] != 4)
        return -1;
    if (checksum(buf + 2, 5) != buf[7])
        return -1;
    *pos = (uint16_t)(buf[5] | (buf[6] << 8));
    return 0;
}
