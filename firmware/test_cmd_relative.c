/* Host test for cmd_relative: reads ops from stdin, prints what the ring does.
 *   i q0..q5   init the ring with a reading
 *   c q0..q5   correction for a reading (kept for the following applies)
 *   a t0..t5   apply a plan target, print the goal
 *   h          hold, print the goal
 */
#include <stdio.h>
#include "cmd_relative.h"

static int read6(int16_t v[6])
{
    for (int j = 0; j < 6; ++j) {
        int x;
        if (scanf("%d", &x) != 1) return -1;
        v[j] = (int16_t)x;
    }
    return 0;
}

static void print6(char tag, const int16_t v[6])
{
    printf("%c", tag);
    for (int j = 0; j < 6; ++j) printf(" %d", v[j]);
    printf("\n");
}

int main(void)
{
    cmd_ring r;
    int16_t v[6], corr[6] = {0}, cmd[6];
    char op;
    while (scanf(" %c", &op) == 1) {
        if (op == 'h') {
            cmd_hold(&r, cmd);
            print6('g', cmd);
            continue;
        }
        if (read6(v) != 0) return 1;
        if (op == 'i') {
            cmd_ring_init(&r, v);
        } else if (op == 'c') {
            cmd_correction(&r, v, corr);
            print6('k', corr);
        } else if (op == 'a') {
            cmd_apply(&r, v, corr, cmd);
            print6('g', cmd);
        } else {
            return 1;
        }
    }
    return 0;
}
