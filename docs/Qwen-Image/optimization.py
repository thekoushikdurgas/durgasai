from typing import Any
from typing import Callable
from typing import ParamSpec
import spaces
import torch
from torch.utils._pytree import tree_map
from spaces.zero.torch.aoti import ZeroGPUCompiledModel, ZeroGPUWeights

P = ParamSpec('P')


TRANSFORMER_IMAGE_SEQ_LENGTH_DIM = torch.export.Dim('image_seq_length')
TRANSFORMER_TEXT_SEQ_LENGTH_DIM = torch.export.Dim('text_seq_length')

TRANSFORMER_DYNAMIC_SHAPES = {
    'hidden_states': {
        1: TRANSFORMER_IMAGE_SEQ_LENGTH_DIM,
    },
    'encoder_hidden_states': {
        1: TRANSFORMER_TEXT_SEQ_LENGTH_DIM,
    },
    'encoder_hidden_states_mask': {
        1: TRANSFORMER_TEXT_SEQ_LENGTH_DIM,
    },
    'image_rotary_emb': ({
        0: TRANSFORMER_IMAGE_SEQ_LENGTH_DIM,
    }, {
        0: TRANSFORMER_TEXT_SEQ_LENGTH_DIM,
    }),
}


INDUCTOR_CONFIGS = {
    'conv_1x1_as_mm': True,
    'epilogue_fusion': False,
    'coordinate_descent_tuning': True,
    'coordinate_descent_check_all_directions': True,
    'max_autotune': True,
    'triton.cudagraphs': True,
}


def optimize_pipeline_(pipeline: Callable[P, Any], *args: P.args, **kwargs: P.kwargs):

    @spaces.GPU(duration=1500)
    def compile_transformer():
        
        # Only capture what the first `transformer_block` sees.
        with spaces.aoti_capture(pipeline.transformer.transformer_blocks[0]) as call:
            pipeline(*args, **kwargs)

        dynamic_shapes = tree_map(lambda t: None, call.kwargs)
        dynamic_shapes |= TRANSFORMER_DYNAMIC_SHAPES

        # Optionally quantize it.
        # quantize_(pipeline.transformer, Float8DynamicActivationFloat8WeightConfig())
        
        # Only export the first transformer block.
        exported = torch.export.export(
            mod=pipeline.transformer.transformer_blocks[0],
            args=call.args,
            kwargs=call.kwargs,
            dynamic_shapes=dynamic_shapes,
        )
        return spaces.aoti_compile(exported, INDUCTOR_CONFIGS)

    compiled = compile_transformer()
    for block in pipeline.transformer.transformer_blocks:
        weights = ZeroGPUWeights(block.state_dict())
        compiled_block = ZeroGPUCompiledModel(compiled.archive_file, weights)
        block.forward = compiled_block
