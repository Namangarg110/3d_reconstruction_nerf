from dataclasses import dataclass
from typing import Literal

from nerfstudio.process_data import equirect_utils, process_data_utils
from nerfstudio.process_data.colmap_converter_to_nerfstudio_dataset import ColmapConverterToNerfstudioDataset
from nerfstudio.utils.rich_utils import CONSOLE
from pathlib import Path


@dataclass
class VideoToNerfstudioDataset(ColmapConverterToNerfstudioDataset):
    """Process videos into a nerfstudio dataset.

    This script does the following:

    1. Converts the video into images and downscales them.
    2. Calculates the camera poses for each image using `COLMAP <https://colmap.github.io/>`_.
    """

    num_frames_target: int = 300
    """Target number of frames to use per video, results may not be exact."""
    percent_radius_crop: float = 1.0
    """Create circle crop mask. The radius is the percent of the image diagonal."""
    matching_method: Literal["exhaustive", "sequential", "vocab_tree"] = "sequential"
    """Feature matching method to use. Vocab tree is recommended for a balance of speed
    and accuracy. Exhaustive is slower but more accurate. Sequential is faster but
    should only be used for videos."""

    def main(self) -> None:
        """Process video into a nerfstudio dataset."""

        summary_log = []
        summary_log_eval = []
        # Convert video to images

        # If we're not dealing with equirects we can downscale in one step.
        summary_log, num_extracted_frames = process_data_utils.convert_video_to_images(
            self.data,
            image_dir=self.image_dir,
            num_frames_target=self.num_frames_target,
            num_downscales=self.num_downscales,
            crop_factor=self.crop_factor,
            verbose=self.verbose,
            image_prefix="frame_train_" if self.eval_data is not None else "frame_",
            keep_image_dir=False,
        )
        # Create mask
        mask_path = process_data_utils.save_mask(
            image_dir=self.image_dir,
            num_downscales=self.num_downscales,
            crop_factor=(0.0, 0.0, 0.0, 0.0),
            percent_radius=self.percent_radius_crop,
        )
        if mask_path is not None:
            summary_log.append(f"Saved mask to {mask_path}")

        # Run Colmap
        if not self.skip_colmap:
            self._run_colmap(mask_path)

        # Export depth maps
        image_id_to_depth_path, log_tmp = self._export_depth()
        summary_log += log_tmp

        summary_log += self._save_transforms(num_extracted_frames, image_id_to_depth_path, mask_path)

        CONSOLE.log("[bold green]:tada: :tada: :tada: All DONE :tada: :tada: :tada:")

        for summary in summary_log:
            CONSOLE.log(summary)

if __name__ == "__main__":
    video_processor = VideoToNerfstudioDataset(
    data=Path('./small.mp4'),
    output_dir=Path('./data/')
    )
    video_processor.main()