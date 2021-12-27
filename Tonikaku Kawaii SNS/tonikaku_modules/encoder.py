from pathlib import Path
from typing import List, Optional, Sequence

import vapoursynth as vs
from vardautomation import (JAPANESE, AudioStream, ChapterStream, 
                            FileInfo, MatroskaXMLChapters, Mux, 
                            RunnerConfig, SelfRunner, X265Encoder, QAACEncoder,
                            VideoStream, Chapter, SoxCutter, FFmpegAudioExtracter)

core = vs.core

class Encoder:
    runner: SelfRunner
    XML_TAG = 'settings/tags_aac.xml'

    def __init__ (self, file: FileInfo, clip: vs.VideoNode,
                 chapter_list: Optional[List[Chapter]] = None,
                 chapter_names: Sequence[str] = ['', ''],
                 chapter_offset: Optional[int] = None) -> None:

        self.file = file
        self.clip = clip
        self.chapter_list = chapter_list
        self.chapter_names = chapter_names
        self.chapter_offset = chapter_offset

        assert self.file.a_src

        self.v_encoder = X265Encoder('settings/x265_settings')
        self.file.set_name_clip_output_ext('.265')
        self.a_extracters = FFmpegAudioExtracter(self.file, track_in=1, track_out=1)
        self.a_cutters = SoxCutter(self.file, track=1)
        self.a_encoders = QAACEncoder(self.file, track=1, xml_tag=self.XML_TAG)

    def run(self) -> None:
        assert self.file.a_enc_cut

        #From LightArrowExE
        if self.chapter_list:
            assert self.file.chapter
            assert self.file.trims_or_dfs

            Path("/chapters").mkdir(parents=True, exist_ok=True)

            if not isinstance(self.chapter_offset, int):
                self.chapter_offset = self.file.trims_or_dfs[0] * -1  # type: ignore

            chapxml = MatroskaXMLChapters(self.file.chapter)
            chapxml.create(self.chapter_list, self.file.clip.fps)
            chapxml.shift_times(self.chapter_offset, self.file.clip.fps)  # type: ignore
            chapxml.set_names(self.chapter_names)
            chapters = ChapterStream(chapxml.chapter_file, JAPANESE)

        muxer = Mux(
            self.file,
            streams=(
                VideoStream(self.file.name_clip_output, 'Encoded by tuilakhanh', JAPANESE),
                AudioStream(self.file.a_enc_cut.set_track(1), 'AAC 2.0', JAPANESE, self.XML_TAG),
                chapters if self.chapter_list else None
            )
        )

        config = RunnerConfig(
            self.v_encoder, None,
            self.a_extracters, self.a_cutters, self.a_encoders,
            muxer
        )

        self.runner = SelfRunner(self.clip, self.file, config)
        self.runner.run()

        self.runner.do_cleanup()