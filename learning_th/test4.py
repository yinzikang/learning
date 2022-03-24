pos = 255
speed = 255
force = 255

command = "091003E8000306090000" + hex(pos)[2:].zfill(2) + hex(speed)[2:].zfill(2) + hex(force)[2:].zfill(2)
data = bytearray.fromhex(command)
crc = 0xFFFF
for idx in data:
    crc ^= idx
    for i in range(8):
        if (crc & 1) != 0:
            crc >>= 1
            crc ^= 0xA001
        else:
            crc >>= 1
crc = "%04X" % (crc)
command = command + crc[2] + crc[3] + crc[0] + crc[1]

print(bytearray.fromhex(command))
