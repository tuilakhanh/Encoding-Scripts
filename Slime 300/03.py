from typing import Optional, Sequence, Union, Tuple

import vapoursynth as vs
from vardautomation import (JAPANESE, FileInfo, PresetBD, PresetAAC, PresetChapXML
                            , VPath, MplsReader)
import lvsfunc as lvf
from slime300_module import enc, flt

core = vs.core


EP_NUM = __file__[-5:-3]

#Import Source
JP_BD = FileInfo(r"D:/BDMV/[BDMV][210630][slime 300][Vol.1]/BD/BDMV/STREAM/00002.m2ts",
                (24, 34069), idx=lambda x: lvf.src(x, force_lsmas=True, cachedir=''),
                 preset=[PresetBD, PresetAAC, PresetChapXML])

JP_NOP = FileInfo(r"D:\BDMV\[BDMV][210630][slime 300][Vol.1]\BD\BDMV\STREAM\00007.m2ts",
                (24, -24), idx=lambda x: lvf.src(x, force_lsmas=True, cachedir=''),)

JP_NED = FileInfo(r"D:\BDMV\[BDMV][210630][slime 300][Vol.1]\BD\BDMV\STREAM\00012.m2ts",
                (24, -24), idx=lambda x: lvf.src(x, force_lsmas=True, cachedir=''),)

JP_BD.do_qpfile = True

#Tạo Chapter
CHAPTERS = MplsReader(r"D:/BDMV/[BDMV][210630][slime 300][Vol.1]/BD", lang=JAPANESE) \
    .get_playlist()[1].mpls_chapters[int(EP_NUM)-1].to_chapters()
CHAP_NAMES: Sequence[Optional[str]] = ['Intro', 'Opening', 'Part A', 'Part B', 'Ending']

OPSTART = 2518
OPEND = 4674

EDSTART = 31887
EDEND = 34044


class Filtering:
    def main(self) -> vs.VideoNode:
        from EoEfunc.denoise import CMDegrain, BM3D
        from vsutil import depth, get_y
        import vardefunc as vdf
        import muvsfunc as muvf
        from G41Fun import MaskedDHA
        from adptvgrnMod import adptvgrnMod

        src = JP_BD.clip_cut
        src = depth(src, 16)
        out = depth(src, 32)

        luma = get_y(out)
        lmask = vdf.mask.FDOG().get_mask(get_y(out), lthr=0.065, hthr=0.065).std.Maximum().std.Minimum()
        lmask = lmask.std.Median().std.Convolution([1] * 9)

        descale = lvf.scale.descale(luma, height=720, kernel=lvf.kernels.Bicubic(b=-0.5, c=0.5), upscaler=None)
        upscale  = vdf.scale.fsrcnnx_upscale(descale, None, height=720*2, shader_file=r"assest/FSRCNNX_x2_56-16-4-1.glsl", profile="zastin")
        downscale = muvf.SSIM_downsample(upscale, src.width, src.height, filter_param_a=0, filter_param_b=0)
        downscale = core.std.MaskedMerge(luma, downscale, lmask)
        scaled = vdf.misc.merge_chroma(downscale, out)
        out = depth(scaled, 16)

        aa = lvf.sraa(out, rfactor=1.6)
        dehalo = MaskedDHA(aa, rx=1.8, ry=1.8, darkstr=0.1, brightstr=1.0, maskpull=48, maskpush=140)
        draken = flt.line_darkening(dehalo, 0.225)
        out = draken

        ref = CMDegrain(out, tr=2, thSAD=150, search=3,contrasharp=True, RefineMotion=True)
        denoise = BM3D(depth(out, 32), sigma=1, radius=1, ref=depth(ref, 32), CUDA=False)
        out = depth(denoise, 16)
        
        credit = flt.restore_credits(
            out, src, (OPSTART, OPEND), (EDSTART, EDEND), show_mask=False,
            ep=src, ncop=JP_NOP.clip_cut, nced=JP_NED.clip_cut, thr=25 << 8, prefilter=True
        )
        out = credit 

        deband = flt.masked_f3kdb(out, rad = 28, thr = 16, grain=24)
        grain = adptvgrnMod(deband, strength=0.2, luma_scaling=10, size=1.25, sharp=80, grain_chroma=False, seed=42069)   

        return depth(grain, 10)


if __name__ == '__main__':
    filtered = Filtering().main()
    brrrr = enc.Encoder(JP_BD, filtered, CHAPTERS, CHAP_NAMES)
    brrrr.run(clean_up=True)
else:
    JP_BD.clip.set_output(0)
    JP_BD.clip_cut.set_output(1)
    FILTERED = Filtering().main()
    if not isinstance(FILTERED, vs.VideoNode):
        for i, clip_filtered in enumerate(FILTERED, start=1):
            clip_filtered.set_output(i)
    else:
        FILTERED.set_output(3)