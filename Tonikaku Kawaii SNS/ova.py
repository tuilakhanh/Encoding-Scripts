from typing import Optional, Sequence
import vapoursynth as vs
from vardautomation import (FileInfo, MplsReader, JAPANESE,
                            PresetAAC, PresetBD, PresetChapXML)
from tonikaku_modules import flt, enc

core = vs.core

JPBD = FileInfo(r"D:/BDMV/[BDMV] TONIKAWA Over The Moon For You OVA/BD_VIDEO/BDMV/STREAM/00002.m2ts",
                (24,-24), preset=[PresetBD, PresetAAC, PresetChapXML])

JP_NOP = FileInfo(r"D:/BDMV/[BDMV][210322][Fly Me to the Moon Blu-Ray Box]/KWXA-2574-1/BDMV/STREAM/00010.m2ts",
                (24,-24))

JP_NED = FileInfo(r"D:/BDMV/[BDMV][210322][Fly Me to the Moon Blu-Ray Box]/KWXA-2574-1/BDMV/STREAM/00011.m2ts",
                (24,-24))

CHAPTERS = MplsReader(r"D:/BDMV/[BDMV] TONIKAWA Over The Moon For You OVA/BD_VIDEO", lang=JAPANESE) \
    .get_playlist()[3].mpls_chapters[0].to_chapters()
CHAP_NAMES: Sequence[Optional[str]] = ['Intro', 'Opening', 'Part A', 'Part B', 'Ending']

OPSTART = 552
EDSTART = 31889

OP_OFFSET = 2
ED_OFFSET = 2

class Filtering:
    def main(self) -> vs.VideoNode:
        from lvsfunc.util import replace_ranges as rfs
        import vardefunc as vdf
        from G41Fun import MaskedDHA
        from vsutil import depth, get_y
        from adptvgrnMod import adptvgrnMod
        import awsmfunc as awf
        from vardautomation import make_comps

        src = JPBD.clip_cut
        out = depth(src, 16)

        op_mask = vdf.dcm(
            src, src[OPSTART:OPSTART+JP_NOP.clip_cut.num_frames-OP_OFFSET], JP_NOP.clip_cut[:-OP_OFFSET],
            start_frame=OPSTART, thr=25, prefilter=True) \
                if OPSTART is not False else get_y(core.std.BlankClip(src))

        ed_mask = vdf.dcm(
            src, src[EDSTART:EDSTART+JP_NED.clip_cut.num_frames-ED_OFFSET], JP_NED.clip_cut[:-ED_OFFSET],
            start_frame=EDSTART, thr=25, prefilter=True) \
                if EDSTART is not False else get_y(core.std.BlankClip(src))
        credit_mask = core.std.Expr([op_mask, ed_mask], expr='x y +')
        credit_mask = depth(credit_mask, 16).std.Binarize()

        scaled, credit_mask1 = flt.rescaler(out, height=844)
        aa = flt.antialiasing(scaled, strength=1.2)
        out = aa
        
        creadit = core.std.MaskedMerge(out, depth(src, 16), credit_mask1)
        creadit = core.std.MaskedMerge(out, depth(src, 16), credit_mask)
        out = creadit

        dehalo = MaskedDHA(out, rx=1.8, ry=1.8, darkstr=0.1, brightstr=1)
        darken = flt.line_darkening(dehalo, 0.175)
        out = darken

        denoise = flt.denoising(out, bm3d_sigma=[1.5, 1, 1])
        out = denoise

        deband = flt.debanding(out)
        grain = adptvgrnMod(deband, strength=0.2, luma_scaling=10, size=1.25, sharp=80, grain_chroma=False, seed=42069)
        out = grain

        crop = core.std.Crop(out, 0, 0, 132, 132)
        crop = awf.bbmod(crop, 0, 1, 0, 1)
        crop = core.std.AddBorders(crop, 0, 0, 132, 132)
        out = rfs(out, crop, [(554, 1448)])

        return depth(out, 10)

if __name__ == '__main__':
    filtered = Filtering().main()
    khanhcc = enc.Encoder(JPBD, filtered, CHAPTERS, CHAP_NAMES)
    khanhcc.run(clean_up=True)
else:
    JPBD.clip.set_output(0)
    FILTERED = Filtering().main()
    if not isinstance(FILTERED, vs.VideoNode):
        for i, clip_filtered in enumerate(FILTERED, start=1):
            clip_filtered.set_output(i)
    else:
        FILTERED.set_output(3)