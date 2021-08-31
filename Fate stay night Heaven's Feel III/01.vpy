from vapoursynth import core, VideoNode, GRAY, YUV
import atomchtools as atf
import fvsfunc as fvf
import kagefunc as kg
import havsfunc
import mvsfunc as mvf
import GradFun3 as gfa
import insaneAA 
import cooldegrain
import finedehalo as fd
import HardAAp2 as hiaa
import descale as dsc
import lvsfunc as lvf



def fate(clip):
    gray = core.std.ShufflePlanes(clip, 0, GRAY)
    desc = insaneAA.revert_upscale(gray, descale_strength=0.28, kernel='bicubic', descale_height=855, bicubic_b=0.33, bicubic_c=0.33)
    graydh = fd.fine_dehalo(desc, darkstr=0, brightstr=0.7, showmask=0, thmi=50, thma=95, rx=2.4, ry=2.4, useMtEdge=True)
    rescale = insaneAA.rescale(graydh, eedi3_mode=0, nnedi3_mode=2, dx=1920, dy=1080, nsize=6, beta=0.40, alpha=0.05, gamma=1) 
    aa = insaneAA.insaneAA(clip, external_aa=rescale, output_mode=0, input_mode=1)
    mixyuv = core.std.ShufflePlanes([aa, clip, clip], [0,1,2], clip.format.color_family)
    return mixyuv
	
core.set_max_cache_size(126000)
episode = core.ffms2.Source(r'fate.mkv')
insaneaa = fate(episode)


#mrgc = episode.std.Trim(0, 527)+insaneaa.std.Trim(528, 6656)+episode.std.Trim(6657, 6846)+insaneaa.std.Trim(6847, 160057)+episode.std.Trim(160058, 166712)+insaneaa.std.Trim(166713, 167628)+episode.std.Trim(167629, 167715)+insaneaa.std.Trim(167716, 168144)+episode.std.Trim(168145, 168503)


clip16 = core.fmtc.bitdepth(insaneaa, bits=16)
clip_y = mvf.GetPlane(clip16, 0)
pf = core.dfttest.DFTTest(clip_y, sigma=7, tbsize=3, opt=3)
den_y = cooldegrain.CoolDegrain(clip_y, tr=1, thsad=26, thsadc=26, bits=16, blksize=8, overlap=4, pf=pf)
filtered = core.std.ShufflePlanes([den_y, clip16, clip16], [0,1,2], colorfamily=clip16.format.color_family)
db = core.f3kdb.Deband(filtered, range=31, y=16, cb=10, cr=10, grainy=22, grainc=22, output_depth=16, keep_tv_range=True)
Mask = atf.retinex_edgemask(clip16, sigma=0.1, draft=False, opencl=True)
Mask2 = core.std.Binarize(Mask,9828,0)
filtered2 = core.std.MaskedMerge(db, clip16, Mask2, planes=[0,1,2], first_plane=True)
filteredmix = core.std.Expr([filtered2, clip16], 'x {val} * y 1 {val} - * +'.format(val=0.26))
sub = core.xyvsf.TextSub(filteredmix, r'III_-_Spring_Song.ass')
episode3 = core.fmtc.bitdepth(sub, bits=10)
episode3.set_output()


















