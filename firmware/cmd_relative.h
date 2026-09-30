/* Latency-aware command-relative servo goals.
 *
 * A position servo holding load settles short of its goal by a steady error
 * (gravity sag). Sending `reading + step` builds every new goal from the
 * short reading, so a loaded joint stops moving once the policy's steps are
 * smaller than the sag: the arm stalls short of the object. Instead apply
 * the policy's step to the goal that produced the reading:
 *
 *     cmd_t = cmd_{t-1-CMD_LAT} + (plan - q_read)
 *
 * which cancels any constant error between goal and reading. CMD_LAT is the
 * number of control ticks between writing a goal and the encoder reading that
 * reflects it, beyond the one tick the loop itself takes; measure it on the
 * arm. Mirrored by pocketpolicy.realworld.CommandRing.
 */
#ifndef CMD_RELATIVE_H
#define CMD_RELATIVE_H
#include <stdint.h>

#ifndef CMD_LAT
#define CMD_LAT 2
#endif
#define CMD_MAX_CORR 114   /* ticks (~10 deg): anti-windup on a blocked joint */
#define CMD_GRIPPER 5      /* commanded absolutely: closing on an object stalls it */

typedef struct {
    int16_t buf[CMD_LAT + 1][6];   /* the last CMD_LAT + 1 goals sent */
    int head;                      /* oldest slot = cmd_{t-1-CMD_LAT} */
} cmd_ring;

/* Fill the ring with the first reading, as if the arm had been holding it. */
void cmd_ring_init(cmd_ring *r, const int16_t q[6]);

/* corr = clamp(cmd_{t-1-CMD_LAT} - q_read), gripper 0. Once per plan. */
void cmd_correction(const cmd_ring *r, const int16_t q_read[6], int16_t corr[6]);

/* cmd = target + corr (saturated to int16), recorded as this tick's goal. */
void cmd_apply(cmd_ring *r, const int16_t target[6], const int16_t corr[6], int16_t cmd[6]);

/* A tick with no new goal: the servos keep the newest one; record it again. */
void cmd_hold(cmd_ring *r, int16_t cmd[6]);

#endif
