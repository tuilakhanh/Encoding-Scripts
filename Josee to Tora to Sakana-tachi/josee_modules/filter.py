from typing import Any, Dict, Optional, Tuple, List, Union

from EoEfunc.denoise import BM3D
import vapoursynth as vs
from vsutil import get_y, depth
import vardefunc as vdf

core = vs.core

def hybrid_denoise(clip: vs.VideoNode, knlm_h: float = 0.5, sigma: float = 2,
                        knlm_args: Optional[Dict[str, Any]] = None, bm3d_args: Optional[Dict[str, Any]] = None) -> vs.VideoNode:
        knargs = dict(a=2, d=3, device_type='gpu', device_id=0, channels='UV')
        if knlm_args is not None:
            knargs.update(knlm_args)

        b3args = dict(radius = 1, profile="fast", CUDA=True)
        if bm3d_args is not None:
            b3args.update(bm3d_args)

        luma = get_y(clip)
        luma = BM3D(luma, sigma=sigma, **b3args)
        chroma = core.knlm.KNLMeansCL(clip, h=knlm_h, **knargs)

        return vdf.misc.merge_chroma(luma, chroma)

def _get_bits(clip: vs.VideoNode, expected_depth: int = 16) -> Tuple[int, vs.VideoNode]:
    """Checks bitdepth, set bitdepth if necessary, and sends original clip's bitdepth back with the clip"""
    from vsutil import depth, get_depth

    bits = get_depth(clip)
    return bits, depth(clip, expected_depth) if bits != expected_depth else clip

def detail_mask(clip: vs.VideoNode,
                sigma: float = 1.0, rxsigma: List[int] = [50, 200, 350],
                pf_sigma: Optional[float] = 1.0,
                rad: int = 3, brz: Tuple[int, int] = (2500, 4500),
                rg_mode: int = 17,
                ) -> vs.VideoNode:
    """
    A detail mask aimed at preserving as much detail as possible within darker areas,
    even if it contains mostly noise.
    """
    from kagefunc import kirsch
    from vsutil import get_y, iterate

    bits, clip = _get_bits(clip)

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
    return rm_grain if bits == 16 else depth(rm_grain, bits)

def masked_f3kdb(clip: vs.VideoNode,
                 rad: int = 16,
                 thr: Union[int, List[int]] = 24,
                 grain: Union[int, List[int]] = [12, 0],
                 mask_args: Dict[str, Any] = {}
                 ) -> vs.VideoNode:
    from debandshit import dumb3kdb

    deb_mask_args: Dict[str, Any] = dict(brz=(1300, 3300))
    deb_mask_args |= mask_args

    bits, clip = _get_bits(clip)

    deband_mask = detail_mask(clip, **deb_mask_args)

    deband = dumb3kdb(clip, radius=rad, threshold=thr, grain=grain)
    deband_masked = core.std.MaskedMerge(deband, clip, deband_mask)
    deband_masked = deband_masked if bits == 16 else depth(deband_masked, bits)
    return deband_masked, deband_mask

def antialiasing(clip: vs.VideoNode, strength: float = 1.2) -> vs.VideoNode:
    from lvsfunc.aa import clamp_aa, nneedi3_clamp, upscaled_sraa

    bits, clip = _get_bits(clip)

    aa_weak = nneedi3_clamp(clip)
    aa_str = upscaled_sraa(clip)
    aa_clamped = clamp_aa(clip, aa_weak, aa_str, strength=strength)
    return aa_clamped if bits == 16 else depth(aa_clamped, bits)

def line_darkening(clip: vs.VideoNode, strength: float = 0.2, **kwargs: Any) -> vs.VideoNode:
    """
    Darken lineart through Toon.
    Taken from varde's repository.
    """
    import havsfunc as haf

    darken = haf.Toon(clip, strength, **kwargs)
    darken_mask = core.std.Expr(
        [core.std.Convolution(clip, [5, 10, 5, 0, 0, 0, -5, -10, -5], divisor=4, saturate=False),
         core.std.Convolution(clip, [5, 0, -5, 10, 0, -10, 5, 0, -5], divisor=4, saturate=False)],
        ['x y max {neutral} / 0.86 pow {peak} *'
            .format(neutral=1 << (clip.format.bits_per_sample-1),  # type: ignore[union-attr]
                    peak=(1 << clip.format.bits_per_sample)-1)])  # type: ignore[union-attr]
    return core.std.MaskedMerge(clip, darken, darken_mask)