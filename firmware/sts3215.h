/* Minimal Feetech STS/SCS bus-servo protocol for the SO-101 follower arm.
 *
 * Packet: 0xFF 0xFF ID LEN INSTR PARAM... CHECKSUM
 *   LEN      = number of params + 2
 *   CHECKSUM = ~(ID + LEN + INSTR + sum(params)) & 0xFF
 * Registers used (STS3215 memory table):
 *   42 Goal Position (2 bytes, little endian)
 *   56 Present Position (2 bytes, little endian)
 */
#ifndef STS3215_H
#define STS3215_H
#include <stdint.h>

#define STS_BROADCAST 0xFE
#define STS_INST_READ 0x02
#define STS_INST_SYNC_WRITE 0x83
#define STS_GOAL_POSITION 42
#define STS_PRESENT_POSITION 56

/* Build a SYNC WRITE of goal positions for n servos. Returns packet length.
   buf must hold 8 + 3*n bytes. Positions are raw encoder counts 0..4095. */
int sts_sync_write_goal(uint8_t *buf, const uint8_t *ids, const uint16_t *pos, int n);

/* Build a READ of the present position of one servo. buf holds 8 bytes. */
int sts_read_position(uint8_t *buf, uint8_t id);

/* Parse a status reply to a position read. Returns 0 and sets *pos on
   success, -1 on a bad header, length or checksum. */
int sts_parse_position(const uint8_t *buf, int len, uint8_t id, uint16_t *pos);

#endif
