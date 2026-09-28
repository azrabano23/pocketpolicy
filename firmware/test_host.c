/* Host test for the servo protocol: prints packets as hex for comparison. */
#include <stdio.h>
#include "sts3215.h"

static void dump(const uint8_t *b, int n)
{
    for (int i = 0; i < n; ++i) printf("%02X", b[i]);
    printf("\n");
}

int main(void)
{
    uint8_t buf[32];
    const uint8_t ids[2] = {1, 2};
    const uint16_t pos[2] = {2048, 1000};
    dump(buf, sts_sync_write_goal(buf, ids, pos, 2));
    dump(buf, sts_read_position(buf, 3));
    uint8_t reply[8] = {0xFF, 0xFF, 3, 4, 0, 0x34, 0x08, 0};
    unsigned s = 3 + 4 + 0 + 0x34 + 0x08;
    reply[7] = (uint8_t)(~s & 0xFF);
    uint16_t p = 0;
    int rc = sts_parse_position(reply, 8, 3, &p);  /* call before reading p */
    printf("%d %u\n", rc, (unsigned)p);
    reply[7] ^= 1;
    printf("%d\n", sts_parse_position(reply, 8, 3, &p));
    return 0;
}
