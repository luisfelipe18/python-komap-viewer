import struct

import sdl2
import glm
from OpenGL.GLUT import *
from OpenGL.GL import *
from OpenGL.GLU import *

TILE_SIZE = 4
PATCH_TILE_SIZE = 8
FLT_MIN = 1.175494351e-38
MAX_PATH = 260

GL_COMPRESSED_RGB_S3TC_DXT1_EXT = 0x83F0
GL_COMPRESSED_RGBA_S3TC_DXT1_EXT = 0x83F1
GL_COMPRESSED_RGBA_S3TC_DXT3_EXT = 0x83F2
GL_COMPRESSED_RGBA_S3TC_DXT5_EXT = 0x83F3
# GL_VERTEX_SHADER = 0x8B31
# GL_COMPILE_STATUS = 0x8B81

_D3DFORMAT = {
    'D3DFMT_UNKNOWN': 0,
    'D3DFMT_DXT1': 827611204,
    'D3DFMT_DXT3': 861165636,
    'D3DFMT_DXT5': 894720068
}


# Define the Input struct
class MyInput:
    def __init__(self):
        self.up = sdl2.SDL_bool()
        self.down = sdl2.SDL_bool()
        self.left = sdl2.SDL_bool()
        self.right = sdl2.SDL_bool()
        self.space = sdl2.SDL_bool()


user_input = MyInput()
pInput = user_input


class _N3TexHeader:
    def __init__(self):
        self.szID: str = ""
        self.nWidth: int = 0
        self.nHeight: int = 0
        self.Format: int = 0
        self.bMipMap: bool = False

    @staticmethod
    def unpack(data):
        header = _N3TexHeader()
        header.szID = data[:4]
        header.nWidth, header.nHeight, header.Format, header.bMipMap = struct.unpack("i i i ?", data[4:])
        return header


class _N3Texture:
    def __init__(self):
        self.m_lpTexture = 0
        self.m_Header = _N3TexHeader()
        self.compTexSize = 0
        self.compTexData = None
        self.m_iLOD = 0
        self.m_szName = None
        self.m_szFileName = None

    def __repr__(self):
        return f"_N3Texture(m_lpTexture={self.m_lpTexture}, " \
               f"m_Header={self.m_Header}, compTexSize={self.compTexSize}, " \
               f"compTexData={self.compTexData}, m_iLOD={self.m_iLOD}, " \
               f"m_szName={self.m_szName}, m_szFileName={self.m_szFileName})"


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


TileDirU = [
    [0.0, 1.0, 0.0, 1.0],  # [down][RT, LT, RB, LB]
    [0.0, 0.0, 1.0, 1.0],  # [left][RT, LT, RB, LB]
    [1.0, 0.0, 1.0, 0.0],  # [up][RT, LT, RB, LB]
    [1.0, 1.0, 0.0, 0.0],  # [right][RT, LT, RB, LB]
    [1.0, 0.0, 1.0, 0.0],  # [down][LT, RT, LB, RB]
    [0.0, 0.0, 1.0, 1.0],  # [left][LT, RT, LB, RB]
    [0.0, 1.0, 0.0, 1.0],  # [up][LT, RT, LB, RB]
    [1.0, 1.0, 0.0, 0.0],  # [right][LT, RT, LB, RB]
]

TileDirV = [
    [0.0, 0.0, 1.0, 1.0],  # [down][RT, LT, RB, LB]
    [1.0, 0.0, 1.0, 0.0],  # [left][RT, LT, RB, LB]
    [1.0, 1.0, 0.0, 0.0],  # [up][RT, LT, RB, LB]
    [0.0, 1.0, 0.0, 1.0],  # [right][RT, LT, RB, LB]
    [0.0, 0.0, 1.0, 1.0],  # [down][LT, RT, LB, RB]
    [0.0, 1.0, 0.0, 1.0],  # [left][LT, RT, LB, RB]
    [1.0, 1.0, 0.0, 0.0],  # [up][LT, RT, LB, RB]
    [1.0, 0.0, 1.0, 0.0],  # [right][LT, RT, LB, RB]
]


def open_ko_file(map_name):
    with open(map_name, 'rb') as ko_filename:
        ko_file = ko_filename.read()

    print('file opened correctly!', end='\n')
    print('')
    return ko_file


def N3LoadTexture(pFile, pTex):
    # NOTE: length of the texture name
    nL = struct.unpack("i", pFile.read(4))[0]

    if nL > 0:
        pTex.m_szName = pFile.read(nL)
        pTex.m_szName = pFile.rstrip(b'\x00')

    pTex.m_Header = _N3TexHeader()
    pTex.m_Header.szID = pFile.read(4).decode('utf-8')
    pTex.m_Header.nWidth = struct.unpack('i', pFile.read(4))[0]
    pTex.m_Header.nHeight = struct.unpack('i', pFile.read(4))[0]
    pTex.m_Header.Format = struct.unpack('i', pFile.read(4))[0]
    pTex.m_Header.bMipMap = bool(struct.unpack('i', pFile.read(4))[0])

    # NOTE: the textures contain multiple mipmap data "blocks"
    if pTex.m_Header.Format == _D3DFORMAT['D3DFMT_DXT1']:
        pTex.compTexSize = pTex.m_Header.nWidth * pTex.m_Header.nHeight // 2
    elif pTex.m_Header.Format == _D3DFORMAT['D3DFMT_DXT3']:
        pTex.compTexSize = pTex.m_Header.nWidth * pTex.m_Header.nHeight
    elif pTex.m_Header.Format == _D3DFORMAT['D3DFMT_DXT5']:
        pTex.compTexSize = pTex.m_Header.nWidth * pTex.m_Header.nHeight * 2
        print("ER: D3DFMT_DXT5 tex; need to verify size!\n\n")
        exit(-1)

    nMMC = 1
    if pTex.m_Header.bMipMap:
        # Calculate the number of MipMap levels
        nW, nH = pTex.m_Header.nWidth, pTex.m_Header.nHeight
        while nW >= 4 and nH >= 4:
            nMMC += 1
            nW //= 2
            nH //= 2
    else:
        # Non-mipmap textures are not implemented
        print("ER: Need to implement non-mipmap textures!")
        exit(-1)

    if pTex.m_iLOD > 0:
        # Skip to the right LOD mipmaps
        print("ER: m_iLOD > 0 need to skip to the right level!")
        exit(-1)

    # Get the first mipmap data for LOD = 0
    if pTex.compTexData is not None:
        # Clean up the previous data
        del pTex.compTexData

    pTex.compTexData = bytearray(pFile.read(pTex.compTexSize))

    # Allocate an OpenGL texture
    pTex.m_lpTexture = glGenTextures(1)

    # Set the texture to unit 0
    glActiveTexture(GL_TEXTURE0)

    # Bind the texture to send data to the GPU
    glBindTexture(GL_TEXTURE_2D, pTex.m_lpTexture)

    # Send the pixels to the GPU
    texFormat = None
    if pTex.m_Header.Format == _D3DFORMAT['D3DFMT_DXT1']:
        texFormat = GL_COMPRESSED_RGBA_S3TC_DXT1_EXT
    elif pTex.m_Header.Format == _D3DFORMAT['D3DFMT_DXT3']:
        texFormat = GL_COMPRESSED_RGBA_S3TC_DXT3_EXT
    elif pTex.m_Header.Format == _D3DFORMAT['D3DFMT_DXT5']:
        texFormat = GL_COMPRESSED_RGBA_S3TC_DXT5_EXT
    else:
        print("ER: Unknown texture format!")
        exit(-1)

    # Upload compressed texture data
    glTexImage2D(
        GL_TEXTURE_2D, 0, texFormat,
        pTex.m_Header.nWidth, pTex.m_Header.nHeight, 0,
        GL_RGBA, GL_UNSIGNED_BYTE, pTex.compTexData
    )

    # Generate mipmaps for scaling
    data = memoryview(pTex.compTexData)

    # Generate mipmaps for scaling
    gluBuild2DMipmaps(
        GL_TEXTURE_2D, GL_RGBA, pTex.m_Header.nWidth, pTex.m_Header.nHeight,
        GL_RGBA, GL_UNSIGNED_BYTE, data
    )


def SetVerts(verts, n):
    # Bind to the array buffer so that we may send our data to the GPU
    glBindBuffer(GL_ARRAY_BUFFER, verBuffer)

    # Send vertex data to the GPU and set as STATIC
    glBufferData(GL_ARRAY_BUFFER, n * ctypes.sizeof(ctypes.c_float), verts, GL_STATIC_DRAW)

    quads = n // 28
    elems = (GLuint * (quads * 6))()

    for i in range(quads):
        elements = [
            0 + i * 4, 1 + i * 4, 2 + i * 4,
            2 + i * 4, 3 + i * 4, 0 + i * 4
        ]

        elems[i * 6: (i + 1) * 6] = elements

    # Bind to the element buffer so that we may send our data to the GPU
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, eleBuffer)

    # Send element data to the GPU and set as STATIC
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, quads * 6 * ctypes.sizeof(GLuint), elems, GL_STATIC_DRAW)


def N3Init():
    # Initialize SDL2 library
    if sdl2.SDL_Init(sdl2.SDL_INIT_EVERYTHING) != 0:
        print("SDL_Init:", sdl2.SDL_GetError())
        exit(-1)

    # Create game window
    global window
    window = sdl2.SDL_CreateWindow(
        b"N3Terrain",
        sdl2.SDL_WINDOWPOS_CENTERED,
        sdl2.SDL_WINDOWPOS_CENTERED,
        1024,
        720,
        sdl2.SDL_WINDOW_OPENGL
    )

    # Check for window
    if window is None:
        print("SDL_CreateWindow:", sdl2.SDL_GetError())
        exit(-1)

    # Create an OpenGL context for the window
    global context
    context = sdl2.SDL_GL_CreateContext(window)

    # Check context
    if context is None:
        print("SDL_GL_CreateContext:", sdl2.SDL_GetError())
        exit(-1)

    # Set the buffer swap interval to get vsync
    if sdl2.SDL_GL_SetSwapInterval(1) != 0:
        print("SDL_GL_SetSwapInterval:", sdl2.SDL_GetError())
        exit(-1)

    # Initialize the OpenGL library function calls
    sdl2.SDL_GL_MakeCurrent(window, context)

    # Enable the depth test
    glEnable(GL_DEPTH_TEST)
    return window, context


def SetTexture(texInd1, texInd2):
    # NOTE: set the texture to unit 0
    glActiveTexture(GL_TEXTURE0)

    # NOTE: bind to the texture so that we may send our data to the GPU
    glBindTexture(GL_TEXTURE_2D, m_pTileTex[texInd1].m_lpTexture)

    # NOTE: bind the uniform "tex" in the fragment shader to the unit 0 texture
    glUniform1i(glGetUniformLocation(shaderProgram, "tex1"), 0)

    # NOTE: not all the tiles have two textures!
    if m_pTileTex[texInd2].m_Header.Format == 0:
        glUniform1i(glGetUniformLocation(shaderProgram, "tex2"), 0)
        return

    # NOTE: set the texture to unit 1
    glActiveTexture(GL_TEXTURE1)

    # NOTE: bind to the texture so that we may send our data to the GPU
    glBindTexture(GL_TEXTURE_2D, m_pTileTex[texInd2].m_lpTexture)

    # NOTE: bind the uniform "tex" in the fragment shader to the unit 1 texture
    glUniform1i(glGetUniformLocation(shaderProgram, "tex2"), 1)


def N3Quit(window, context):
    # Free the OpenGL context
    sdl2.SDL_GL_DeleteContext(context)
    context = None

    # Free the SDL2 window
    sdl2.SDL_DestroyWindow(window)
    window = None

    # Quit the SDL2 library
    sdl2.SDL_Quit()


if __name__ == '__main__':

    window, context = N3Init()

    fpMap = open_ko_file('karus_start.gtd')

    if fpMap is None:
        print('Error: File not found!', end='\n')
        os.system('exit')

    # NOTE: get the size of the map
    m_ti_MapSize = struct.unpack("i", fpMap[:4])[0]
    print(f'DB: m_ti_MapSize = {m_ti_MapSize}', end='\n')

    # NOTE: the number of patches which exist within the map
    m_pat_MapSize = int((m_ti_MapSize - 1) / PATCH_TILE_SIZE)
    print(f'DB: m_pat_MapSize = {m_pat_MapSize}', end='\n')

    print('')

    # NOTE: read in the map data
    total_elements = m_ti_MapSize * m_ti_MapSize
    total_bytes = 8 * total_elements
    m_pMapData = [
        _N3MapData().create(fpMap[ind:ind + 8]) for ind in range(4, total_bytes, 8)
    ]
    print("DB: m_pMapData[0] = {")
    m_pMapData[0].to_print()
    print(f'total elements: {len(m_pMapData)}\n')

    # NOTE: read in the middle Y and radius for each patch
    m_ppPatchMiddleY = []
    m_ppPatchRadius = []
    total_bytes = total_bytes + 4

    for x in range(0, m_pat_MapSize):
        v_ = [
            struct.unpack("f", fpMap[ind_start:ind_start + 4])[0]
            for ind_start in range(total_bytes, total_bytes + 2 * 4 * m_pat_MapSize, 4)
        ]
        total_bytes = total_bytes + 2 * 4 * m_pat_MapSize
        m_ppPatchMiddleY.append(v_[::2])
        m_ppPatchRadius.append(v_[1::2])
    print(
        f"DB: m_ppPatchMiddleY[0][0] = {m_ppPatchMiddleY[0][0]}\n"
        f"DB: m_ppPatchRadius[0][0] = {m_ppPatchRadius[0][0]}\n\n"
    )

    # NOTE: read in the grass attributes
    m_pGrassAttr = bytearray(fpMap[total_bytes:total_bytes + m_ti_MapSize * m_ti_MapSize])
    total_bytes += m_ti_MapSize * m_ti_MapSize
    print(f"DB: m_pGrassAttr[0] = {m_pGrassAttr[0]}")

    # NOTE: "set" the grass number
    m_pGrassNum = bytearray([5] * (m_ti_MapSize * m_ti_MapSize))
    print(f"DB: m_pGrassNum[0] = {m_pGrassNum[0]}\n")

    # NOTE: read in the grass file name
    m_pGrassFileName = fpMap[total_bytes:total_bytes + MAX_PATH].decode('latin-1').rstrip('\0')
    total_bytes += MAX_PATH
    print(f"DB: m_pGrassFileName = \"{m_pGrassFileName}\"\n")

    # NOTE: if there isn't a grass filename then zero out the grass attr
    if m_pGrassFileName == "":
        m_pGrassAttr = bytearray([0] * (m_ti_MapSize * m_ti_MapSize))
    else:
        # CN3Terrain::LoadGrassInfo(void)
        pass

    # Start CN3Terrain::LoadTileInfo(hFile)

    m_NumTileTex = struct.unpack("i", fpMap[total_bytes:total_bytes + 4])[0]
    total_bytes += 4
    print(f"DB: m_NumTileTex = {m_NumTileTex}")

    # NOTE: load in all the textures needed for the terrain
    m_pTileTex = [_N3Texture() for _ in range(m_NumTileTex)]
    # print(f"DB: m_pTileTex[0] = {_N3Texture()}")  # Assuming _N3Texture has a __repr__ method

    # NOTE: load in the number of texture files needed
    NumTileTexSrc = struct.unpack("i", fpMap[total_bytes:total_bytes + 4])[0]
    total_bytes += 4
    print(f"DB: NumTileTexSrc = {NumTileTexSrc}\n")

    # NOTE: load in all the texture file names
    SrcName = []
    for i in range(NumTileTexSrc):
        name = fpMap[total_bytes:total_bytes + MAX_PATH].decode('latin-1').rstrip('\0')
        name = name.split('.gtt')[0] + '.gtt'
        total_bytes += MAX_PATH
        SrcName.append(name)

    print(f"DB: SrcName[(NumTileTexSrc-1)] = \"{SrcName[NumTileTexSrc - 1]}\"\n")

    # NOTE: SrcIdx is the index to the filename
    # NOTE: TileIdx is the index to the texture within the file (that's why
    # we have to skip)

    # NOTE: associate the textures with their tile texture objects
    hTTGFile = None
    SrcIdx = 0
    TileIdx = 0

    for i in range(m_NumTileTex):
        SrcIdx = struct.unpack('h', fpMap[total_bytes:total_bytes + 2])[0]
        total_bytes += 2
        TileIdx = struct.unpack('h', fpMap[total_bytes:total_bytes + 2])[0]
        total_bytes += 2
        if i == (m_NumTileTex - 1):
            print(f"DB: [i==(m_NumTileTex-1)]\nDB: SrcIdx = {SrcIdx}, TileIdx = {TileIdx}")

        # Read the texture file
        try:
            hTTGFile = open(SrcName[SrcIdx], 'rb')
        except FileNotFoundError:
            print(f"ER: Cannot load texture: \"{SrcName[SrcIdx]}\"")
            os.system("pause")
            break

        if i == (m_NumTileTex - 1):
            print(f"DB: SrcName[{SrcIdx}] = \"{SrcName[SrcIdx]}\"\n")

        for j in range(TileIdx):
            nL = struct.unpack('i', hTTGFile.read(4))[0]

            m_szName = b''
            if nL > 0:
                m_szName = hTTGFile.read(nL)
                m_szName = m_szName.rstrip(b'\x00')

            HeaderOrg = _N3TexHeader()
            HeaderOrg.szID = hTTGFile.read(4).decode('utf-8')
            HeaderOrg.nWidth = struct.unpack('i', hTTGFile.read(4))[0]
            HeaderOrg.nHeight = struct.unpack('i', hTTGFile.read(4))[0]
            HeaderOrg.Format = struct.unpack('i', hTTGFile.read(4))[0]
            HeaderOrg.bMipMap = bool(struct.unpack('i', hTTGFile.read(4))[0])

            if (
                    HeaderOrg.Format == _D3DFORMAT['D3DFMT_DXT1'] or
                    HeaderOrg.Format == _D3DFORMAT['D3DFMT_DXT3'] or
                    HeaderOrg.Format == _D3DFORMAT['D3DFMT_DXT5']
            ):
                # Skipping textures for tiles within the file that aren't the tile we need for m_pTileTex[i]
                iSkipSize = 0
                iWTmp = HeaderOrg.nWidth
                iHTmp = HeaderOrg.nHeight

                if HeaderOrg.bMipMap:
                    while iWTmp >= 4 and iHTmp >= 4:
                        if HeaderOrg.Format == _D3DFORMAT['D3DFMT_DXT1']:
                            iSkipSize += iWTmp * iHTmp // 2
                        else:
                            iSkipSize += iWTmp * iHTmp
                        iWTmp //= 2
                        iHTmp //= 2

                    iWTmp = HeaderOrg.nWidth // 2
                    iHTmp = HeaderOrg.nHeight // 2
                    while iWTmp >= 4 and iHTmp >= 4:
                        iSkipSize += iWTmp * iHTmp * 2
                        iWTmp //= 2
                        iHTmp //= 2
                else:
                    if HeaderOrg.Format == _D3DFORMAT['D3DFMT_DXT1']:
                        iSkipSize += HeaderOrg.nWidth * HeaderOrg.nHeight // 2
                    else:
                        iSkipSize += HeaderOrg.nWidth * HeaderOrg.nHeight
                    iSkipSize += HeaderOrg.nWidth * HeaderOrg.nHeight * 2
                    if HeaderOrg.nWidth >= 1024:
                        iSkipSize += 256 * 256 * 2

                hTTGFile.seek(iSkipSize, os.SEEK_CUR)
            else:
                print("ER: Unsupported Texture format!\n\n")
                os.system("pause")

        m_pTileTex[i].m_iLOD = 0
        N3LoadTexture(hTTGFile, m_pTileTex[i])

        hTTGFile.close()

    # Loop for freeing memory of SrcName
    del SrcName

    # Read NumLightMap
    NumLightMap = struct.unpack('<i', fpMap[total_bytes: total_bytes + 4])[0]
    total_bytes += 4
    print("DB: NumLightMap =", NumLightMap, "\n")

    if NumLightMap > 0:
        print("ER: NumLightMap > 0! Need to implement this.\n")
        exit(-1)

    # Load River
    m_iRiverCount = struct.unpack('<i', fpMap[total_bytes: total_bytes + 4])[0]
    total_bytes += 4
    print("DB: m_iRiverCount =", m_iRiverCount, "\n")

    if m_iRiverCount > 0:
        print("ER: m_iRiverCount > 0! Need to implement this.\n")
        exit(-1)

    # Load Pond
    m_iPondMeshNum = struct.unpack('<i', fpMap[total_bytes: total_bytes + 4])[0]
    total_bytes += 4
    print("DB: m_iPondMeshNum =", m_iPondMeshNum, "\n")

    if m_iPondMeshNum > 0:
        print("WR: m_iPondMeshNum > 0! Ignoring this...\n")

    # End CN3Terrain::LoadTileInfo(hFile)
    del fpMap

    # Source code for the vertex shader
    vertSource = b"""
    #version 110

    attribute vec3 pos;
    attribute vec2 texcoord1;
    attribute vec2 texcoord2;

    varying vec2 fragTexcoord1;
    varying vec2 fragTexcoord2;

    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 proj;

    void main() {
        fragTexcoord1 = texcoord1;
        fragTexcoord2 = texcoord2;
        gl_Position = proj * view * model * vec4(pos, 1.0);
    }
    """

    # Allocate vertex shader program
    vertShader = glCreateShader(GL_VERTEX_SHADER)

    # Load the vertex shader's source code
    glShaderSource(vertShader, [vertSource])

    # Compile the vertex shader's source code
    glCompileShader(vertShader)

    # Get the status of the compilation
    status = glGetShaderiv(vertShader, GL_COMPILE_STATUS)

    # Check if the compilation was successful
    if status != GL_TRUE:
        print("Error: Vertex shader compilation failed")
        print(glGetShaderInfoLog(vertShader))  # Print the shader compilation error log if available
        exit(-1)

    # Source code for the fragment shader
    fragSource = b'''
    #version 110

    varying vec2 fragTexcoord1;
    varying vec2 fragTexcoord2;
    uniform sampler2D tex1;
    uniform sampler2D tex2;

    void main() {
        vec4 t0 = texture2D(tex1, fragTexcoord1) * vec4(1.0, 1.0, 1.0, 1.0);
        vec4 t1 = texture2D(tex2, fragTexcoord2) * vec4(1.0, 1.0, 1.0, 1.0);
        if (t0 == t1)
            gl_FragColor = t0;
        else
            gl_FragColor = (t0 + t1);
    }
    '''

    # Allocate fragment shader program
    fragShader = glCreateShader(GL_FRAGMENT_SHADER)

    # Load the fragment shader's source code
    glShaderSource(fragShader, [fragSource])

    # Compile the fragment shader's source code
    glCompileShader(fragShader)

    # Get the status of the compilation
    status = glGetShaderiv(fragShader, GL_COMPILE_STATUS)
    # Check if the compilation was successful
    if status != GL_TRUE:
        print("Error: Fragment shader compilation failed")
        print(glGetShaderInfoLog(vertShader))  # Print the fragment compilation error log if available
        exit(-1)

    # Create a shader program out of the vertex and fragment shaders
    shaderProgram = glCreateProgram()

    # Attach the vertex and fragment shaders
    glAttachShader(shaderProgram, vertShader)
    glAttachShader(shaderProgram, fragShader)

    # Link the shader program
    glLinkProgram(shaderProgram)

    status = GLint(0)
    # Get the status of linking the program
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, status)

    # If the program failed to link, print the error
    if status == GL_FALSE:
        buffer = ctypes.create_string_buffer(512)
        glGetProgramInfoLog(shaderProgram, 512, None, buffer)
        print("glLinkProgram: ")
        print(buffer.value.decode())
        exit(-1)

    # Use the newly compiled shader program
    glUseProgram(shaderProgram)

    # Allocate an array buffer on the GPU
    verBuffer = GLuint(0)
    glGenBuffers(1, ctypes.byref(verBuffer))

    # Bind to the array buffer so that we may send our data to the GPU
    glBindBuffer(GL_ARRAY_BUFFER, verBuffer)

    # Get a pointer to the position attribute variable in the shader program
    posAttrib = glGetAttribLocation(shaderProgram, "pos")

    # Specify the stride (spacing) and offset for array buffer which will be used in place of the attribute variable in the shader program
    glVertexAttribPointer(posAttrib, 3, GL_FLOAT, GL_FALSE, 7 * ctypes.sizeof(ctypes.c_float), 0)

    # Enable the attribute
    glEnableVertexAttribArray(posAttrib)

    # Get a pointer to the texcoord attributes in the shader program
    texAttrib1 = glGetAttribLocation(shaderProgram, "texcoord1")
    texAttrib2 = glGetAttribLocation(shaderProgram, "texcoord2")

    # Specify the stride (spacing) and offset for array buffer which will be used in place of the attribute variable in the shader program
    glVertexAttribPointer(texAttrib1, 2, GL_FLOAT, GL_FALSE, 7 * ctypes.sizeof(ctypes.c_float),
                          ctypes.c_void_p(3 * ctypes.sizeof(ctypes.c_float)))
    glVertexAttribPointer(texAttrib2, 2, GL_FLOAT, GL_FALSE, 7 * ctypes.sizeof(ctypes.c_float),
                          ctypes.c_void_p(5 * ctypes.sizeof(ctypes.c_float)))

    # Enable the attributes
    glEnableVertexAttribArray(texAttrib1)
    glEnableVertexAttribArray(texAttrib2)

    # Allocate a GPU buffer for the element data
    eleBuffer = GLuint(0)
    glGenBuffers(1, ctypes.byref(eleBuffer))

    # Push all the terrain information to the GPU to save on draw time
    gl_buf_offsets = (ctypes.c_int * (m_ti_MapSize * m_ti_MapSize))()
    buffer_len = 28 * (m_ti_MapSize * m_ti_MapSize)
    vertex_info = (ctypes.c_float * buffer_len)()
    gl_os = 0
    offset = 0

    for pX in range(1, m_ti_MapSize - 1):
        for pZ in range(1, m_ti_MapSize - 1):
            if pX < 1 or pX > (m_ti_MapSize - 1):
                continue
            if pZ < 1 or pZ > (m_ti_MapSize - 1):
                continue

            MapData = m_pMapData[pX * m_ti_MapSize + pZ]

            u10, u11, u12, u13 = TileDirU[MapData.Tex1Dir]
            v10, v11, v12, v13 = TileDirV[MapData.Tex1Dir]
            u20, u21, u22, u23 = TileDirU[MapData.Tex2Dir]
            v20, v21, v22, v23 = TileDirV[MapData.Tex2Dir]

            x0 = pX * TILE_SIZE
            y0 = m_pMapData[pX * m_ti_MapSize + pZ].fHeight
            z0 = pZ * TILE_SIZE

            x1 = pX * TILE_SIZE
            y1 = m_pMapData[pX * m_ti_MapSize + pZ + 1].fHeight
            z1 = (pZ + 1) * TILE_SIZE

            x2 = (pX + 1) * TILE_SIZE
            y2 = m_pMapData[(pX + 1) * m_ti_MapSize + pZ + 1].fHeight
            z2 = (pZ + 1) * TILE_SIZE

            x3 = (pX + 1) * TILE_SIZE
            y3 = m_pMapData[(pX + 1) * m_ti_MapSize + pZ].fHeight
            z3 = pZ * TILE_SIZE

            vertices = (ctypes.c_float * 28)(
                x0, y0, z0, u10, v10, u20, v20,  # Top-left
                x1, y1, z1, u11, v11, u21, v21,  # Top-right
                x2, y2, z2, u12, v12, u22, v22,  # Bottom-right
                x3, y3, z3, u13, v13, u23, v23  # Bottom-left
            )

            n = len(vertices)

            gl_buf_offsets[(pX * m_ti_MapSize) + pZ] = offset
            for i in range(n):
                vertex_info[offset] = vertices[i]
                offset += 1

    SetVerts(vertex_info, offset)

    # TESTING
    # ========================================================================

    model = glm.mat4(1.0)
    angle = glm.radians(3.6)  # Equivalent to (float)M_PI / 50.0f
    model = glm.rotate(model, angle, glm.vec3(0.0, 1.0, 0.0))

    uniModel = glGetUniformLocation(shaderProgram, "model")
    glUniformMatrix4fv(uniModel, 1, GL_FALSE, glm.value_ptr(model))

    pDist = 155.0
    view = glm.lookAt(
        glm.vec3(pDist, pDist, pDist),
        glm.vec3(0.0, 0.0, 0.0),
        glm.vec3(0.0, 1.0, 0.0)
    )

    uniView = glGetUniformLocation(shaderProgram, "view")
    glUniformMatrix4fv(uniView, 1, GL_FALSE, glm.value_ptr(view))

    proj = glm.perspective(glm.radians(45.0), 800.0 / 600.0, 0.5, 1000.0)

    uniProj = glGetUniformLocation(shaderProgram, "proj")
    glUniformMatrix4fv(uniProj, 1, GL_FALSE, glm.value_ptr(proj))

    m_position = glm.vec3(106.96, 80.0, 206.13)
    m_direction = glm.vec3(0.929, 0.0, 0.368)

    # END TESTING
    # ========================================================================

    while user_input.type != sdl2.SDL_QUIT:
        while sdl2.SDL_PollEvent(sdl2.pointer(user_input)) != 0:
            if user_input.type == sdl2.SDL_KEYUP:
                if user_input.key.keysym.sym == sdl2.SDLK_UP:
                    pInput.up = sdl2.SDL_FALSE
                elif user_input.key.keysym.sym == sdl2.SDLK_DOWN:
                    pInput.down = sdl2.SDL_FALSE
                elif user_input.key.keysym.sym == sdl2.SDLK_LEFT:
                    pInput.left = sdl2.SDL_FALSE
                elif user_input.key.keysym.sym == sdl2.SDLK_RIGHT:
                    pInput.right = sdl2.SDL_FALSE
                elif user_input.key.keysym.sym == sdl2.SDLK_SPACE:
                    pInput.space = sdl2.SDL_FALSE
            elif user_input.type == sdl2.SDL_KEYDOWN:
                if user_input.key.keysym.sym == sdl2.SDLK_UP:
                    pInput.up = sdl2.SDL_TRUE
                elif user_input.key.keysym.sym == sdl2.SDLK_DOWN:
                    pInput.down = sdl2.SDL_TRUE
                elif user_input.key.keysym.sym == sdl2.SDLK_LEFT:
                    pInput.left = sdl2.SDL_TRUE
                elif user_input.key.keysym.sym == sdl2.SDLK_RIGHT:
                    pInput.right = sdl2.SDL_TRUE
                elif user_input.key.keysym.sym == sdl2.SDLK_SPACE:
                    pInput.space = sdl2.SDL_TRUE

        if pInput.up and not pInput.space:
            m_position += m_direction
        if pInput.down and not pInput.space:
            m_position -= m_direction

        if pInput.left:
            m_direction = glm.rotate(m_direction, angle, glm.vec3(0.0, 1.0, 0.0))

        if pInput.right:
            m_direction = glm.rotate(m_direction, -angle, glm.vec3(0.0, 1.0, 0.0))

        if pInput.space and pInput.up:
            m_position.y += 1.0
        if pInput.space and pInput.down:
            m_position.y -= 1.0

        view = glm.lookAt(m_position, m_position + m_direction, glm.vec3(0.0, 1.0, 0.0))
        uniView = glGetUniformLocation(shaderProgram, "view")
        glUniformMatrix4fv(uniView, 1, GL_FALSE, glm.value_ptr(view))

        # ===

        # NOTE: clear the screen buffer
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # ===

        lastPatZ = 0
        lastPatX = 0

        curPatZ = int(m_position.z / (TILE_SIZE * PATCH_TILE_SIZE))
        curPatX = int(m_position.x / (TILE_SIZE * PATCH_TILE_SIZE))

        for pX in range(curPatX - 8, curPatX + 8):
            for pZ in range(curPatZ - 8, curPatZ + 8):
                if pX < 1 or pX > (m_pat_MapSize - 1):
                    continue
                if pZ < 1 or pZ > (m_pat_MapSize - 1):
                    continue

                m_ti_LBPoint_x = pX * PATCH_TILE_SIZE
                m_ti_LBPoint_z = pZ * PATCH_TILE_SIZE

                for ix in range(PATCH_TILE_SIZE):
                    for iz in range(PATCH_TILE_SIZE):
                        tx = ix + m_ti_LBPoint_x
                        tz = iz + m_ti_LBPoint_z

                        MapData = m_pMapData[(tx * m_ti_MapSize) + tz]

                        SetTexture(MapData.Tex1Idx, MapData.Tex2Idx)

                        i1 = gl_buf_offsets[(tx * m_ti_MapSize) + tz]
                        i2 = i1 // 28
                        i3 = i2 * 6

                        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, ctypes.c_void_p(i3 * sizeof(GLuint)))

            # NOTE: swap the front and back buffers
        sdl2.SDL_GL_SwapWindow(window)

    N3Quit(window, context)
