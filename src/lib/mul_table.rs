// Mul table for 1 / (radius*radius)
pub(crate) const MUL_TABLE_DOUBLE: [i32; 320] = [
    512, 512, /* < radius 1 starts */ 512, 227, 256, 327, 227, 334, 256, 404, 327, 270, 227,
    387, 334, 291, 256, 226, 404, 363, 327, 297, 270, 247, 227, 209, 387, 359, 334, 311, 291, 272,
    256, 240, 226, 213, 202, 382, 363, 344, 327, 311, 297, 283, 270, 258, 247, 237, 227, 218, 209,
    201, 387, 373, 359, 346, 334, 322, 311, 301, 291, 281, 272, 264, 256, 248, 240, 233, 226, 220,
    213, 208, 202, 196, 191, 186, 181, 176, 172, 168, 319, 311, 304, 297, 290, 283, 277, 270, 264,
    258, 253, 247, 242, 237, 232, 227, 222, 218, 213, 209, 205, 201, 197, 193, 190, 186, 183, 179,
    176, 173, 170, 167, 164, 161, 317, 311, 306, 301, 296, 291, 286, 281, 277, 272, 268, 264, 260,
    256, 252, 248, 244, 240, 237, 233, 230, 226, 223, 220, 217, 213, 210, 208, 205, 202, 199, 196,
    194, 191, 188, 367, 363, 358, 353, 349, 344, 340, 336, 331, 327, 323, 319, 315, 311, 308, 304,
    300, 297, 293, 290, 286, 283, 280, 277, 273, 270, 267, 264, 261, 258, 256, 253, 250, 247, 245,
    242, 239, 237, 234, 232, 229, 227, 225, 222, 220, 218, 216, 213, 211, 209, 207, 205, 203, 201,
    199, 197, 195, 193, 192, 190, 188, 186, 184, 183, 369, 366, 362, 359, 356, 353, 349, 346, 343,
    340, 337, 334, 331, 328, 325, 322, 319, 317, 314, 311, 309, 306, 303, 301, 298, 296, 293, 291,
    288, 286, 284, 281, 279, 277, 274, 272, 270, 268, 266, 264, 262, 260, 258, 256, 254, 252, 250,
    248, 246, 244, 242, 240, 238, 237, 235, 233, 231, 230, 228, 226, 225, 223, 221, 220, 218, 217,
    215, 213, 212, 210, 209, 208, 206, 205, 203, 202, 200, 199, 198, 196, 195, 194, 192, 191, 190,
    188, 187, 186, 185, 183, 182, 181, 180, 179, 178, 176, 175, 174, 173, 172, 171, 170, 169, 168,
    166, 165, 164,
];

// Shift right table for 1 / (radius*radius)
pub(crate) const SHR_TABLE_DOUBLE: [i32; 320] = [
    9, 9, /* < radius 1 starts */ 11, 11, 12, 13, 13, 14, 14, 15, 15, 15, 15, 16, 16, 16, 16,
    16, 17, 17, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19,
    19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 21, 21, 21,
    21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
    21, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22,
    22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
    23, 23, 23, 23, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
    24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
    24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
    24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
    24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
];

// Mul table for 1 / (radius*2)
pub(crate) const MUL_TABLE_TWICE_RAD: [i32; 256] = [
    512, 32768, 16384, 10922, 8192, 6553, 5461, 4681, 4096, 3640, 3276, 2978, 2730, 2520, 2340,
    2184, 2048, 1927, 1820, 1724, 1638, 1560, 1489, 1424, 1365, 1310, 1260, 1213, 1170, 1129, 1092,
    1057, 1024, 992, 963, 936, 910, 885, 862, 840, 819, 799, 780, 762, 744, 728, 712, 697, 682,
    668, 655, 642, 630, 618, 606, 595, 585, 574, 564, 555, 546, 537, 528, 520, 512, 504, 496, 489,
    481, 474, 468, 461, 455, 448, 442, 436, 431, 425, 420, 414, 409, 404, 399, 394, 390, 385, 381,
    376, 372, 368, 364, 360, 356, 352, 348, 344, 341, 337, 334, 330, 327, 324, 321, 318, 315, 312,
    309, 306, 303, 300, 297, 295, 292, 289, 287, 284, 282, 280, 277, 275, 273, 270, 268, 266, 264,
    262, 260, 258, 256, 254, 252, 250, 248, 246, 244, 242, 240, 239, 237, 235, 234, 232, 230, 229,
    227, 225, 224, 222, 221, 219, 1736, 1724, 1713, 1702, 1691, 1680, 1669, 1659, 1648, 1638, 1628,
    1618, 1608, 1598, 1588, 1579, 1569, 1560, 1551, 1542, 1533, 1524, 1515, 1506, 1497, 1489, 1481,
    1472, 1464, 1456, 1448, 1440, 1432, 1424, 1416, 1409, 1401, 1394, 1387, 1379, 1372, 1365, 1358,
    1351, 1344, 1337, 1330, 1323, 1317, 1310, 1304, 1297, 1291, 1285, 1278, 1272, 1266, 1260, 1254,
    1248, 1242, 1236, 1230, 1224, 1219, 1213, 1208, 1202, 1197, 1191, 1186, 1180, 1175, 1170, 1165,
    1159, 1154, 1149, 1144, 1139, 1134, 1129, 1125, 1120, 1115, 1110, 1106, 1101, 1096, 1092, 1087,
    1083, 1078, 1074, 1069, 1065, 1061, 1057, 1052, 2097, 2088, 2080, 2072, 2064, 2056, 2048,
];

// Shift right table for 1 / (2*radius)
pub(crate) const SHR_TABLE_TWICE_RAD: [i32; 256] = [
    9, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
    16, 16, 16, 16, 16, 16, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
    19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
    19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
    19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
    19, 19, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 20,
];

pub(crate) const MUL_TABLE_STACK_BLUR: [i32; 255] = [
    512, 512, 456, 512, 328, 456, 335, 512, 405, 328, 271, 456, 388, 335, 292, 512, 454, 405, 364,
    328, 298, 271, 496, 456, 420, 388, 360, 335, 312, 292, 273, 512, 482, 454, 428, 405, 383, 364,
    345, 328, 312, 298, 284, 271, 259, 496, 475, 456, 437, 420, 404, 388, 374, 360, 347, 335, 323,
    312, 302, 292, 282, 273, 265, 512, 497, 482, 468, 454, 441, 428, 417, 405, 394, 383, 373, 364,
    354, 345, 337, 328, 320, 312, 305, 298, 291, 284, 278, 271, 265, 259, 507, 496, 485, 475, 465,
    456, 446, 437, 428, 420, 412, 404, 396, 388, 381, 374, 367, 360, 354, 347, 341, 335, 329, 323,
    318, 312, 307, 302, 297, 292, 287, 282, 278, 273, 269, 265, 261, 512, 505, 497, 489, 482, 475,
    468, 461, 454, 447, 441, 435, 428, 422, 417, 411, 405, 399, 394, 389, 383, 378, 373, 368, 364,
    359, 354, 350, 345, 341, 337, 332, 328, 324, 320, 316, 312, 309, 305, 301, 298, 294, 291, 287,
    284, 281, 278, 274, 271, 268, 265, 262, 259, 257, 507, 501, 496, 491, 485, 480, 475, 470, 465,
    460, 456, 451, 446, 442, 437, 433, 428, 424, 420, 416, 412, 408, 404, 400, 396, 392, 388, 385,
    381, 377, 374, 370, 367, 363, 360, 357, 354, 350, 347, 344, 341, 338, 335, 332, 329, 326, 323,
    320, 318, 315, 312, 310, 307, 304, 302, 299, 297, 294, 292, 289, 287, 285, 282, 280, 278, 275,
    273, 271, 269, 267, 265, 263, 261, 259,
];

pub(crate) const SHR_TABLE_STACK_BLUR: [i32; 255] = [
    9, 11, 12, 13, 13, 14, 14, 15, 15, 15, 15, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 17, 18, 18,
    18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20,
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 21, 21, 21,
    21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22,
    22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22,
    22, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
    24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
    24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
    24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
];
