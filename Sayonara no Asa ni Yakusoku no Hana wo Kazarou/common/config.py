from typing import List, Union

import vapoursynth as vs
from vardautomation import (JAPANESE, AudioStream, BasicTool,
                            ChapterStream, FileInfo, MatroskaXMLChapters, Mux,
                            Patch, RunnerConfig, SelfRunner,
                            VideoStream, X265Encoder, FlacEncoder)
from vardautomation.types import Range

core = vs.core


class Encoding:
    runner: SelfRunner
    xml_tag: str = 'xml_tag.xml'

    def __init__(self, file: FileInfo, clip: vs.VideoNode) -> None:
        self.file = file
        self.clip = clip
        assert self.file.a_src


        self.v_encoder = X265Encoder('common/x265_settings')
        self.a_extracters = [
            BasicTool('eac3to', [self.file.path.to_str(), '2:', self.file.a_src.format(1).to_str(), '-log=NUL'])
        ]
        self.a_encoders = [FlacEncoder(self.file, track=2)]

    def run(self, *, do_chaptering: bool = True) -> None:
        assert self.file.a_enc_cut

        muxer = Mux(
            self.file,
            streams=(
                VideoStream(self.file.name_clip_output, 'BDRip by TPN', JAPANESE),
                [AudioStream(self.file.a_enc_cut.format(1), 'FLAC 2.0', JAPANESE)],
                ChapterStream(self.file.chapter, JAPANESE) if do_chaptering and self.file.chapter else None
            )
        )
        # muxer = Mux(self.file)

        config = RunnerConfig(
            self.v_encoder, None,
            self.a_extracters, self.a_cutters, self.a_encoders,
            muxer
        )


        self.runner = SelfRunner(self.clip, self.file, config)
        self.runner.run()

    def do_patch(self, ranges: Union[Range, List[Range]]) -> None:
        p = Patch(self.v_encoder, self.clip, self.file, ranges)
        p.run()
        p.do_cleanup()

    def cleanup(self) -> None:
        assert self.file.chapter
        self.runner.do_cleanup(self.file.chapter, self.xml_tag)

    def chaptering(self, offset: int):
        assert self.file.chapter

        chap = MatroskaXMLChapters(self.file.chapter)
        chap.copy(self.file.name + '_tmp.xml')
        chap.shift_times(offset, self.clip.fps)
        self.file.chapter = chap.chapter_file
