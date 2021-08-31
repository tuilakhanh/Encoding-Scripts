import atomchtools as atf
import insaneAA
import finedehalo as fd
import lvsfunc as lvf
import cooldegrain
from vsutil import depth
from vapoursynth import core, VideoNode, GRAY, YUV

def Kimetsu_aa(clip):
    gray = core.std.ShufflePlanes(clip, 0, GRAY)
    desc = insaneAA.revert_upscale(gray, descale_strength=0.26, kernel='bicubic', descale_height=855, bicubic_b=0.33, bicubic_c=0.33) 
    graydh = fd.fine_dehalo(desc, darkstr=0, brightstr=0.7, showmask=0, thmi=50, thma=95, rx=2.4, ry=2.4, useMtEdge=True)
    rescale = insaneAA.rescale(graydh, eedi3_mode=0, nnedi3_mode=2, dx=1920, dy=1080, nsize=6)   
    aa = insaneAA.insaneAA(clip, external_aa=rescale, output_mode=0, input_mode=1)
    mixyuv = core.std.ShufflePlanes([aa, clip, clip], [0,1,2], clip.format.color_family)
    return mixyuv  

src = lvf.src(r"kimetsu.dgi")
src = depth(src, 16)
aa = Kimetsu_aa(src)

mrgc = aa.std.Trim(0, 7637) + src.std.Trim(7638, 7883) + aa.std.Trim(7884, 42318) +  src.std.Trim(42319, 42631) + aa.std.Trim(42632, 106213) + src.std.Trim(106214, 106736) + aa.std.Trim(106737, 161728) + src.std.Trim(161729, 168023)

clip16 = depth(mrgc, 16)
pf = core.dfttest.DFTTest(clip16, sigma=1, tbsize=3, opt=3, planes=[0,1,2])
den = cooldegrain.CoolDegrain(clip16, tr=1, thsad=24, thsadc=18, bits=16, blksize=8, overlap=4, pf=pf)
db = core.f3kdb.Deband(den, range=31, y=14, cb=11, cr=11, grainy=22, grainc=22, output_depth=16, keep_tv_range=True)
Mask = atf.retinex_edgemask(clip16, sigma=0.1, draft=False, opencl=False)
Mask2 = core.std.Binarize(Mask,9828,0)
filtered = core.std.MaskedMerge(db, clip16, Mask2, planes=[0,1,2], first_plane=True)

sub = core.vsfm.TextSubMod(filtered, r'Kimetsu no Yaiba Movie.ass')

out = depth(sub, 10)
out.set_output(0)

#en = lvf.src(r'kimetsu1.hevc')[79800:80900]
#en.set_output(1)