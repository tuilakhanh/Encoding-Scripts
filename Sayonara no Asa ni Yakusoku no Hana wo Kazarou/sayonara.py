from typing import Optional, Dict, Any, Tuple, List
import vapoursynth as vs
import atomchtools as atf
from vardautomation import FileInfo, PresetBD, make_comps, PresetFLAC, PresetChapXML
from vsutil import get_y, depth, iterate
import vardefunc as vdf
import havsfunc as hvf
core = vs.core
import xvs
from adptvgrnMod import adptvgrnMod
from common import Encoding

JPBD = FileInfo(
    r"D:\[BDMV][181026] さよならの朝に約束の花をかざろう (特装限定版)\BDROM\BDMV\STREAM\00005.m2ts",
    Preset=[PresetBD, PresetChapXML, PresetFLAC]
)

class Filtering:
    def main(self) -> vs.VideoNode:
        src = JPBD.clip
        out = depth(src, 16)

        sharp = xvs.ssharp(out, mask=True)
        sharp = core.std.Merge(out, sharp, 0.4)
        merge = vdf.misc.merge_chroma(sharp, out)
        out = depth(merge, 16)

        ref = hvf.SMDegrain(get_y(out), tr=1, thSAD=300, prefilter=3, search=3, contrasharp=True, RefineMotion=True)
        denoise = self.hybrid_denoise(
            depth(out, 32), knlm_h=0.25, sigma=1.25,
            knlm_args=dict(d=3, a=2, s=3), bm3d_args=dict(ref=depth(ref, 32))
        )
        out = depth(denoise, 16)

        mask = self.detail_mask(out, brz=(1500, 3500))
        deband = vdf.deband.dumb3kdb(out,28, [12, 10, 10], 16, output_depth=16, keep_tv_range=True)
        deband = core.std.MaskedMerge(deband, out, mask)
        out = deband

        grain = adptvgrnMod(out, strength=0.4, luma_scaling=12, size=1.25, sharp=80, grain_chroma=False)
        out = grain

        #sub = core.xyvsf.TextSub()

        return depth(grain, 10)
    
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

    @staticmethod
    def detail_mask(clip: vs.VideoNode,
                    sigma: float = 1.0, rxsigma: List[int] = [50, 200, 350],
                    pf_sigma: Optional[float] = 1.0,
                    rad: int = 3, brz: Tuple[int, int] = (2500, 4500),
                    rg_mode: int = 17,
                    ) -> vs.VideoNode:
        """
        A detail mask aimed at preserving as much detail as possible
        within darker areas, even if it contains mostly noise.
        """

        clip_y = get_y(clip)
        pf = core.bilateral.Gaussian(clip_y, sigma=pf_sigma) if pf_sigma else clip_y
        ret = core.retinex.MSRCP(pf, sigma=rxsigma, upper_thr=0.005)

        blur_ret = core.bilateral.Gaussian(ret, sigma=sigma)
        blur_ret_diff = core.std.Expr([blur_ret, ret], "x y -")
        blur_ret_dfl = core.std.Deflate(blur_ret_diff)
        blur_ret_ifl = iterate(blur_ret_dfl, core.std.Inflate, 4)
        blur_ret_brz = core.std.Binarize(blur_ret_ifl, brz[0])
        blur_ret_brz = core.morpho.Close(blur_ret_brz, size=8)

        kirsch_mask = kirsch(clip_y).std.Binarize(brz[1])
        kirsch_ifl = kirsch_mask.std.Deflate().std.Inflate()
        kirsch_brz = core.std.Binarize(kirsch_ifl, brz[1])
        kirsch_brz = core.morpho.Close(kirsch_brz, size=4)

        merged = core.std.Expr([blur_ret_brz, kirsch_brz], "x y +")
        rm_grain = core.rgvs.RemoveGrain(merged, rg_mode)
        return depth(rm_grain, 16)

if __name__ == '__main__':

    filtered = Filtering().main()
    khanhcc = Encoding(JPBD, filtered)
    khanhcc.chaptering(0-JPBD.frame_start)  # type: ignore
    khanhcc.run()
    khanhcc.cleanup()
else:
    JPBD.clip_cut.set_output(0)
    FILTERED = Filtering().main()
    if not isinstance(FILTERED, vs.VideoNode):
        for i, clip_filtered in enumerate(FILTERED, start=1):
            clip_filtered.set_output(i)
    else:
        FILTERED.set_output(3)


        
#make_comps(clips=dict(src=src, filter=out), force_bt709 = True, slowpics = True, collection_name = "Sayonara comps", frames = [7632], public = False)

