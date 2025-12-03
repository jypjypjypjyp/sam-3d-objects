from typing import Literal, Union
import numpy as np
from loguru import logger
from PIL import Image


from .trellis_image_to_3d import TrellisImageTo3DPipeline
from sam3d_objects.model.backbone.tdfy_dit.utils import postprocessing_utils


class InferencePipelineTrellis(TrellisImageTo3DPipeline):
    def merge_image_and_mask(
        self,
        image: Union[np.ndarray, Image.Image],
        mask: Union[np.ndarray, Image.Image],
    ):
        if isinstance(image, Image.Image):
            image = np.array(image)
        mask = (np.array(mask) * 255).astype(np.uint8)
        if mask.ndim == 2:
            mask = mask[..., None]
        image = np.concatenate([image[..., :3], mask], axis=-1)
        image = Image.fromarray(image, mode="RGBA")
        return image

    def postprocess_slat_output(
        self, outputs, with_mesh_postprocess, with_texture_baking, use_vertex_color
    ):
        # GLB files can be extracted from the outputs
        logger.info(
            f"Postprocessing mesh with option with_mesh_postprocess {with_mesh_postprocess}, with_texture_baking {with_texture_baking}..."
        )
        if "mesh" in outputs:
            glb = postprocessing_utils.to_glb(
                outputs["gaussian"][0],
                outputs["mesh"][0],
                # Optional parameters
                simplify=0.95,  # Ratio of triangles to remove in the simplification process
                texture_size=1024,  # Size of the texture used for the GLB
                verbose=False,
                with_mesh_postprocess=with_mesh_postprocess,
                with_texture_baking=with_texture_baking,
                use_vertex_color=use_vertex_color,
                rendering_engine="pytorch3d",
            )
        else:
            glb = None

        outputs["glb"] = glb
        return outputs

    def run(
        self,
        images: list[np.ndarray],
        masks: list[np.ndarray],
        num_samples: int = 1,
        seed: int = 42,
        with_mesh_postprocess: bool = True,
        with_texture_baking: bool = True,
        use_vertex_color: bool = False,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: list[str] = ['mesh', 'gaussian'],
        preprocess_image: bool = True,
        mode: Literal['stochastic', 'multidiffusion'] = 'stochastic',
    ) -> dict:
        images = [self.merge_image_and_mask(img, mask) for img, mask in zip(images, masks)]
        outputs = self.run_multi_image(
            images,
            num_samples=num_samples,
            seed=seed,
            sparse_structure_sampler_params=sparse_structure_sampler_params,
            slat_sampler_params=slat_sampler_params,
            formats=formats,
            preprocess_image=preprocess_image,
            mode=mode,
        )
        outputs = self.postprocess_slat_output(
            outputs,
            with_mesh_postprocess,
            with_texture_baking,
            use_vertex_color,
        )
        return outputs
