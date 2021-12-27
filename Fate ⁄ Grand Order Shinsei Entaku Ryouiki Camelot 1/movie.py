import vapoursynth as vs
from typing import Optional, Dict, Any
import atomchtools as atf
import muvsfunc as mvf
import vardefunc as vdf
import havsfunc as hvf
import kagefunc as kgf
from vsutil import depth, get_y, join, plane, get_w
from vardautomation import FileInfo, PresetBD, PresetAAC, PresetChapXML
from common import Encoding
import lvsfunc as lvf
import debandshit as dbs
from finedehalo import fine_dehalo
import xvs

core = vs.core

"""
from vardautomation import UNDEFINED, MatroskaXMLChapters, MplsReader
reader = MplsReader(r"D:\劇場版 Fate_Grand Order -神聖円卓領域キャメロット- 前編 Wandering_ Agateram", UNDEFINED)
reader.write_playlist('chapters', chapters_obj=MatroskaXMLChapters)
exit()
"""
JPBD = FileInfo(
    r"BDMV\STREAM\00000.m2ts", 0, -0,
    preset=[PresetBD, PresetAAC, PresetChapXML]
)


class Filtering:
    def main(self) -> vs.VideoNode:
        """VapouSynth Filter"""

        src = JPBD.clip
        out = depth(src, 16)

        dehalo = fine_dehalo(out, rx=2.4, ry=2.2, darkstr=0, brightstr=1)
        l_mask = lvf.util.quick_resample(dehalo, kgf.retinex_edgemask)
        cwarp = core.std.MaskedMerge(xvs.WarpFixChromaBlend(dehalo, depth=6), out, l_mask)
        cwarp = core.rgvs.Repair(cwarp, dehalo, 13)
        out = depth(cwarp, 16)

        ref = hvf.SMDegrain(get_y(out), tr=1)
        denoise = self.hybrid_denoise(
            depth(out, 32), knlm_h=0.2, sigma=1,
            knlm_args=dict(d=1, a=3, s=3), bm3d_args=dict(ref=depth(ref, 32))
        )
        denoise = depth(denoise, 16)
        out = denoise

        mask = atf.retinex_edgemask(out, sigma=0.1, draft=False).std.Binarize(9828,0).rgvs.RemoveGrain(3).std.Inflate()
        deband = dbs.dumb3kdb(out, 26, [16, 12, 12], 10, output_depth=16, keep_tv_range=True)
        deband = core.std.MaskedMerge(deband, out, mask)
        out = deband

        grain = kgf.adaptive_grain(out, 0.1, luma_scaling=4)
        out = grain

        return depth(out, 10)

    @staticmethod
    def hybrid_denoise(clip: vs.VideoNode, knlm_h: float = 0.5, sigma: float = 2,
                       knlm_args: Optional[Dict[str, Any]] = None, bm3d_args: Optional[Dict[str, Any]] = None) -> vs.VideoNode:
        knargs = dict(a=2, d=3, device_type='gpu', device_id=0, channels='UV')
        if knlm_args is not None:
            knargs.update(knlm_args)

        b3args = dict(radius = 1)
        if bm3d_args is not None:
            b3args.update(bm3d_args)

        luma = get_y(clip)
        luma = atf.BM3DCUDA(luma, sigma=sigma, **b3args)
        chroma = core.knlm.KNLMeansCL(clip, h=knlm_h, **knargs)

        return vdf.misc.merge_chroma(luma, chroma)                                                            

if __name__ == '__main__':

    filtered = Filtering().main()
    khanhcc = Encoding(JPBD, filtered)
    khanhcc.run()
    
else:
    JPBD.clip_cut.set_output(0)
    FILTERED = Filtering().main()
    if not isinstance(FILTERED, vs.VideoNode):
        for i, clip_filtered in enumerate(FILTERED, start=1):
            clip_filtered.set_output(i)
    else:
        FILTERED.set_output(3)
