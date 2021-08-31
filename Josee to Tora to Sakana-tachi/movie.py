from typing import Sequence, Optional

from josee_modules.filter import antialiasing
import vapoursynth as vs
from vardautomation import FileInfo, PresetBD, PresetFLAC, PresetChapXML, MplsReader, JAPANESE
from josee_modules import flt, enc

core = vs.core

JPBD = FileInfo("D:/BDMV/[BDMV][210825][Josee, the Tiger and the Fish][MOVIE]/JOSEE_THE_TIGER_AND_THE_FISH/BDMV/STREAM/00003.m2ts", (0, 141239), preset=[PresetBD, PresetFLAC, PresetChapXML])
NF = FileInfo(r"F:/Khanh de/Josee/[GST] Josee, the Tiger and the Fish (Josee to Tora to Sakana-tachi) [Netflix 1080p].mkv", (11, None))

CHAPTERS = MplsReader(r"D:/BDMV/[BDMV][210825][Josee, the Tiger and the Fish][MOVIE]/JOSEE_THE_TIGER_AND_THE_FISH", lang=JAPANESE) \
    .get_playlist()[1].mpls_chapters[0].to_chapters()
CHAP_NAMES: Sequence[Optional[str]] = ['Part A']

class Filtering:
    def main(self) -> vs.VideoNode:
        import awsmfunc as awf
        from adptvgrnMod import adptvgrnMod
        from EoEfunc.denoise import CMDegrain
        import havsfunc as haf
        import vardefunc as vdf
        from vsutil import depth, get_y

        src = JPBD.clip
        src_crop =  core.std.Crop(src, 0, 0, 138, 138)

        src_NF = NF.clip_cut.std.AssumeFPS(fpsnum=24000, fpsden=1001)
        src_NF = core.std.Crop(src_NF, 0, 0, 138, 138)

        mask_border = core.std.BlankClip(src, height=802, format=vs.GRAY8).std.AddBorders(bottom=2, color=255)
        merge = core.std.MaskedMerge(src_crop, src_NF, mask_border)
        dirty_lines = awf.bbmod(merge, 0, 2, 1, 1)
        out = depth(dirty_lines, 16)

        ref = CMDegrain(get_y(out), tr=2, thSAD=200, search=3, contrasharp=True, RefineMotion=True)
        denoise = flt.hybrid_denoise(
            depth(out, 32), knlm_h=0.15, sigma=1.15,
            knlm_args=dict(d=2, a=2, s=3), bm3d_args=dict(ref=depth(ref, 32))
        )
        out = depth(denoise, 16)

        lmask = vdf.mask.FDOG().get_mask(get_y(out), lthr=0.065, hthr=0.065).std.Maximum().std.Minimum()
        lmask = lmask.std.Median().std.Convolution([1] * 9)
        aa = antialiasing(out, strength=1.6)
        aa_masked = core.std.MaskedMerge(out, aa, lmask)
        out = vdf.util.replace_ranges(aa_masked, out, [133491, 139952])

        dehalo = haf.FineDehalo(out, rx=1.8, ry=1.6, darkstr=0.05, brightstr=1.0)
        darken = flt.line_darkening(dehalo, 0.2).warp.AWarpSharp2(depth=2)
        out = darken

        deband, debm = flt.masked_f3kdb(out, 20, [24, 18, 18], [24, 0])
        grain = adptvgrnMod(deband, strength=0.2, luma_scaling=10, size=1.35, sharp=80, grain_chroma=False, seed=42069)
        out = grain

        #comp = make_comps(dict(Source=src_crop, Filter=out), force_bt709=True, slowpics=True, collection_name="Josee", public=False)

        return depth(out, 10)



if __name__ == '__main__':
    filtered = Filtering().main()
    khanhcc = enc.Encoder(JPBD, filtered, CHAPTERS, CHAP_NAMES)
    khanhcc.run(clean_up=True)
else:
    JPBD.clip.set_output(0)
    NF.clip_cut.set_output(5)
    FILTERED = Filtering().main()
    if not isinstance(FILTERED, vs.VideoNode):
        for i, clip_filtered in enumerate(FILTERED, start=1):
            clip_filtered.set_output(i)
    else:
        FILTERED.set_output(3)