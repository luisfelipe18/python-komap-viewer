import os
import struct

PATCH_TILE_SIZE = 8
FLT_MIN = 1.175494351e-38
MAX_PATH = 260

class _N3Texture:
    pass

class _N3MapData:
    def __init__(self):
        self.fHeight = FLT_MIN
        self.bIsTileFull = True
        self.fHeight = FLT_MIN
        self.Tex1Idx = 1023
        self.Tex1Dir = 0
        self.Tex2Idx = 1023
        self.Tex2Dir = 0

    def create(self, byte_array):
        self.fHeight, bitField = struct.unpack('<fI', byte_array)
        self.bIsTileFull = bitField & 1 == 1
        self.Tex1Idx = (bitField >> 11) & 0b111111111
        self.Tex1Dir = (bitField >> 1) & 0b11111
        self.Tex2Idx = bitField >> 21
        self.Tex2Dir = (bitField >> 6) & 0b11111
        return self

    def to_print(self):
        print(f"DB:     fHeight     = {self.fHeight}")
        print(f"DB:     bIsTileFull = {self.bIsTileFull}")
        print(f"DB:     Tex1Dir     = {self.Tex1Dir}")
        print(f"DB:     Tex2Dir     = {self.Tex2Dir}")
        print(f"DB:     Tex1Idx     = {self.Tex1Idx}")
        print(f"DB:     Tex2Idx     = {self.Tex2Idx}")
        print("DB: }", end='\n')


def open_ko_file(name):
    with open(name, 'rb') as ko_filename:
        ko_file = ko_filename.read()

    print('file opened correctly!', end='\n')
    print('')
    return ko_file


# with open("karus_start.gtd", "rb") as fpMap:
#     # NOTE: get the size of the map
#     m_ti_MapSize = 0
#     m_ti_MapSize = int.from_bytes(fpMap.read(4), byteorder='little')
#     print(f"DB: m_ti_MapSize = {m_ti_MapSize}\n")
#
#     # NOTE: the number of patches which exist within the map
#     m_pat_MapSize = (m_ti_MapSize - 1) // PATCH_TILE_SIZE
#     print(f"DB: m_pat_MapSize = {m_pat_MapSize}\n")
#
#     # NOTE: read in the mapdata
#     m_pMapData = bytearray(fpMap.read(m_ti_MapSize*m_ti_MapSize*28))
#     # Convert bytearray to a list of _N3MapData objects
#     m_pMapData = [_N3MapData.from_buffer(m_pMapData[i*28:(i+1)*28]) for i in range(m_ti_MapSize*m_ti_MapSize)]


if __name__ == '__main__':
    fpMap = open_ko_file('karus_start.gtd')

    if fpMap is None:
        print('Error: File not found!', end='\n')
        os.system('exit')

    # NOTE: get the size of the map

    m_ti_MapSize = struct.unpack("i", fpMap[:4])[0]  # Equivalent to fread(&m_ti_MapSize, 4, 1 ,fpMap)
    print(f'DB: m_ti_MapSize = {m_ti_MapSize}', end='\n')
    # NOTE: the number of patches which exist within the map
    m_pat_MapSize = int((m_ti_MapSize - 1) / PATCH_TILE_SIZE)
    print(f'DB: m_pat_MapSize = {m_pat_MapSize}', end='\n')
    print('')

    # NOTE: read in the map data
    total_elements = m_ti_MapSize * m_ti_MapSize
    total_bytes = 8 * total_elements
    m_pMapData = [
        _N3MapData().create(fpMap[ind:ind + 8]) for it, ind in
        zip(
            range(m_ti_MapSize * m_ti_MapSize), range(4, total_bytes, 8)
        )
    ]
    print("DB: m_pMapData[0] = {")
    m_pMapData[0].to_print()
    print(f'total elementos: {len(m_pMapData)}\n')

    # NOTE: read in the middle Y and radius for each patch
    m_ppPatchMiddleY = []
    m_ppPatchRadius = []
    total_bytes = total_bytes + 4
    for x in range(0, m_pat_MapSize):
        v_ = [
            struct.unpack("f", fpMap[ind_start:ind_start+4])[0]
            for ind_start in range(total_bytes, total_bytes+2*4*m_pat_MapSize, 4)
        ]
        total_bytes = total_bytes+2*4*m_pat_MapSize
        m_ppPatchMiddleY.append(v_[::2])
        m_ppPatchRadius.append(v_[1::2])
    print(
        f"DB: m_ppPatchMiddleY[0][0] = {m_ppPatchMiddleY[0][0]}\n"
        f"DB: m_ppPatchRadius[0][0] = {m_ppPatchRadius[0][0]}\n\n"

    )

    # NOTE: read in the grass attributes
    m_pGrassAttr = []
    for ind in range(total_bytes, total_bytes + m_ti_MapSize*m_ti_MapSize):
        m_pGrassAttr.append(struct.unpack("B", fpMap[ind:ind+1])[0])

    total_bytes = total_bytes + m_ti_MapSize*m_ti_MapSize

    # NOTE: "set" the grass number
    m_pGrassNum = [5 for i in m_pGrassAttr]

    # NOTE: read in the grass file name
    m_pGrassFileName = ''
    total_bytes = total_bytes + MAX_PATH

    if m_pGrassFileName == '':
        m_pGrassAttr = [0 for ind in m_pGrassAttr]
    else:
        print('CN3Terrain::LoadGrassInfo(void)')

    m_NumTileTex = struct.unpack("i", fpMap[total_bytes:total_bytes+4])[0]
    print(f"DB: m_NumTileTex = {m_NumTileTex}\n")

    # m_pTileTex =