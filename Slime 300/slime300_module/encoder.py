from pathlib import Path
from typing import List, Optional, Sequence, Union, Any

from bvsfunc.util.AudioProcessor import video_source
import vapoursynth as vs
from vardautomation import (JAPANESE, AudioStream, ChapterStream, 
                            FileInfo, MatroskaXMLChapters, Mux, 
                            RunnerConfig, SelfRunner, X265Encoder,
                            VideoStream, VPath,Chapter)

core = vs.core

XML_TAG = 'settings/tags_aac.xml'

def resolve_trims(trims: Any) -> Any:
    """Convert list[tuple] into list[list]. begna pls"""
    if all(isinstance(trim, tuple) for trim in trims):
        return [list(trim) for trim in trims]
    return trims


class Encoder:
    """"Regular encoding class"""
    def __init__(self, file: FileInfo, clip: vs.VideoNode,
                 chapter_list: Optional[List[Chapter]] = None,
                 chapter_names: Sequence[str] = ['', ''],
                 chapter_offset: Optional[int] = None) -> None:
        self.file = file
        self.clip = clip
        self.chapter_list = chapter_list
        self.chapter_names = chapter_names
        self.chapter_offset = chapter_offset

    def run(self, clean_up: bool = True) -> None:
        assert self.file.a_src
        assert self.file.a_enc_cut

        v_encoder = X265Encoder('settings/x265_settings')
        a_extracter = Eac3toAudioExtracter(self.file, track_in=2, track_out=1)

        audio_files = video_source(self.file.path.to_str(),
                                   trim_list=resolve_trims(self.file.trims_or_dfs),
                                   trims_framerate=self.file.clip.fps,
                                   flac=False, aac=True, silent=False)

        audio_tracks: List[AudioStream] = []
        for track in audio_files:
            audio_tracks += [AudioStream(VPath(track), 'AAC 2.0', JAPANESE, XML_TAG)]

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
                VideoStream(self.file.name_clip_output, 'Encoded by TuiLaKhanh', JAPANESE),
                audio_tracks, chapters if self.chapter_list else None
            )
        )

        config = RunnerConfig(v_encoder, None, None, None, None, muxer)

        runner = SelfRunner(self.clip, self.file, config)
        runner.run()

        if clean_up:
            runner.do_cleanup()
    