# USB-NCM camera transport contract

The camera is a USB Ethernet/NCM board, not a UVC webcam. The operating system must
expose a working network interface before this SDK can connect. An app cannot
install an Ethernet driver or assign the system interface's static IP.

| Setting | Reference firmware value |
| --- | --- |
| Phone/PC USB interface | 192.168.50.1/30, mask 255.255.255.252 |
| Board | 192.168.50.2 |
| JPEG TCP stream | 5000 |
| Diagnostic UDP discovery | 5001 |
| Discovery request | ASCII `JL-CAMERA-DISCOVER` |

Arrange the USB IP using supported OS adapter settings or board DHCP/firmware.
`hostAddress: ''` permits OS route selection only if the route already works.
Do not use the Wi-Fi address as the USB host binding. The default explicit bind
keeps this connection on the USB network. Discovery is optional and unicast;
streaming can work even if discovery is unsupported. Some board firmware allows
only one client: disconnect the PC reference before attaching/testing the phone.

## JLIP v1 framing

TCP is a byte stream: reads can fragment a header/payload or contain many packets.
The parser reassembles them, bounds buffering, checks CRC32, and keeps only the
newest usable JPEG. All multibyte header integers are **big endian**:

| Offset | Bytes | Value |
| --- | --- | --- |
| 0 | 4 | ASCII `JLIP` |
| 4 | 1 | Version 1 |
| 5 | 1 | Packet type |
| 6 | 2 | Header length 24 |
| 8 | 4 | Sequence, uint32 |
| 12 | 4 | Payload length, max 204800 bytes |
| 16 | 4 | IEEE CRC32 of payload |
| 20 | 4 | Board timestamp in ms, uint32 |

Types: `0x02` hello metadata, `0x03` heartbeat, `0x10` JPEG.
Reply to each heartbeat with a `0x03` packet (not `0x04`), using the same framing.
JPEG markers and dimensions are checked before decoding; only dimensions 32..8192
and at most 12 million pixels are accepted. Malformed input is never a face hit.
Sequence gaps are diagnostic and board timestamp clocks are not synchronized
with the phone. Frame age uses the local monotonic receipt time.

The client reconnects after closure/network errors and a silent socket timeout.
Network status `connected` precedes the first frame. Stop/disconnect must clear
face state, cancel pending processing, and suppress late results from that
session. The included SDK and tests implement those rules.
