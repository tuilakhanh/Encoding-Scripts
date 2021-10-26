import vapoursynth as vs
from vardautomation import FileInfo, PresetBD, X265Encoder, make_comps

core = vs.core

#################################################################
#                             _`				                #
#                          _ooOoo_				                #
#                         o8888888o				                #
#                         88" . "88				                #
#                         (| -_- |)				                #
#                         O\  =  /O				                #
#                      ____/`---'\____				            #
#                    .'  \\|     |//  `.			            #
#                   /  \\|||  :  |||//  \			            #
#                  /  _||||| -:- |||||_  \			            #
#                  |   | \\\  -  /'| |   |			            #
#                  | \_|  `\`---'//  |_/ |			            #
#                  \  .-\__ `-. -'__/-.  /			            #
#                ___`. .'  /--.--\  `. .'___			        #
#             ."" '<  `.___\_<|>_/___.' _> \"".			        #
#            | | :  `- \`. ;`. _/; .'/ /  .' ; |		        #
#            \  \ `-.   \_\_`. _.'_/_/  -' _.' /		        #
#=============`-.`___`-.__\ \___  /__.-'_.'_.-'=================#
#                           `=--=-'                    
#           _.-/`)
#          // / / )
#       .=// / / / )
#      //`/ / / / /
#     // /     ` /
#    ||         /
#     \\       /
#      ))    .'
#     //    /
#          /

JPBD = FileInfo(r"D:/BDMV/Fireworks 2017 1080p JPN Blu-ray AVC DTS-HD MA 5.1-UnperceivedExistence（升起的烟花，从下面看？还是从侧面看？）/FIREWORKS/BDMV/STREAM/00000.m2ts",
                (None, None), preset=[PresetBD])

class Filtering:
    def main(self) -> vs.VideoNode:
        import lvsfunc as lvf
        from vsutil import depth, get_y
        import vardefunc as vdf
        from muvsfunc import SSIM_downsample
        import EoEfunc as eoe
        from ccd import ccd
        import havsfunc as haf
        from adptvgrnMod import adptvgrnMod
        from uchiage_module import filters as flt

        src = JPBD.clip_cut
        src = depth(src, 16)

        denoise = eoe.denoise.BM3D(src, sigma=1.2, profile='fast', CUDA=False, chroma=False)
        denoise = ccd(denoise, 6)
        csharp = eoe.misc.ContraSharpening(denoise, src)
        out = depth(csharp, 32)

        luma = get_y(out)
        lmask = vdf.mask.FDOG().get_mask(get_y(luma), lthr=0.056, hthr=0.056).std.Maximum().std.Minimum()
        lmask = lmask.std.Median().std.Convolution([1] * 9)    #From Varde
        descale = lvf.kernels.Bicubic().descale(luma, 1280, 720)
        upscale_nnedi = vdf.scale.nnedi3cl_double(descale, pscrn=1)
        upscale_nnedi = depth(SSIM_downsample(upscale_nnedi, src.width, src.height, filter_param_a=0, filter_param_b=0), 32)
        upscale_fsrcnnx = vdf.scale.fsrcnnx_upscale(descale, shader_file="assest\FSRCNNX_x2_56-16-4-1.glsl")
        upscale = core.average.Mean([upscale_nnedi, upscale_fsrcnnx])
        scaled = core.std.MaskedMerge(luma, upscale, lmask)
        scaled = vdf.misc.merge_chroma(scaled, out)
        out = depth(scaled, 16)

        aa = lvf.aa.nneedi3_clamp(out, strength=1.6, opencl=False)
        dehalo = haf.FineDehalo(aa, rx=1.6, ry=1.6, darkstr=0.1, brightstr=1.0)
        out = dehalo

        creditsm = lvf.scale.descale_detail_mask(luma, lvf.kernels.Bicubic().scale(descale, 1920, 1080), threshold=0.055)
        creditsm = core.std.Expr(creditsm, 'x 65535 *', vs.GRAY16)
        credits = core.std.MaskedMerge(out, csharp, creditsm)
        out = credits

        deband = flt.masked_f3kdb(out, 18, [28, 25], [24, 0])
        grain = adptvgrnMod(deband, strength=0.25, luma_scaling=14, sharp=80, static=True, grain_chroma=False, seed=22205)
        out = grain
        
        sub = core.xyvsf.TextSub(out, "Uchi.ass")
        out = sub

        # comp = make_comps({
        #         "Source":src,
        #         "Filterd":out,
        #         "Encoded":enc,
        #         "Beatrice":beatrice},
        #         force_bt709=True, slowpics=True, collection_name="Hanabi", public=False)

        return depth(out, 10)

if __name__ == '__main__':
    Filter = Filtering().main()
    X265Encoder('settings\x265_settings').run_enc(Filter, JPBD)
else:
    JPBD.clip_cut.set_output(0)
    Filter = Filtering().main()
    if not isinstance(Filter, vs.VideoNode):
        for i, clip_Filter in enumerate(Filter, start=1):
            clip_Filter.set_output(i)
    else:
        Filter.set_output(3)