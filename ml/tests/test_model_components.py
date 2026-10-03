"""Small controllable neural regressions, without downloading or building Hiera."""
import pytest

torch = pytest.importorskip("torch")

from app.models.sam2_stack import MemoryBank, MemorySlot, MemoryStack
from app.models.sam2_stack.mask_decoder import MaskDecoder


def lightweight_stack():
    stack = MemoryStack.__new__(MemoryStack)
    torch.nn.Module.__init__(stack)
    stack.hidden_dim, stack.mem_dim, stack.num_maskmem = 256, 64, 7
    stack.max_object_pointers = 16
    stack.object_pointers_enabled = True
    stack.temporal_position_encoding_enabled = True
    stack.enable_temporal_pos_encoding_for_object_pointers = False
    stack.no_object_pointer = torch.nn.Parameter(torch.zeros(1, 256))
    stack.memory_temporal_positional_encoding = torch.nn.Parameter(torch.ones(7, 1, 1, 64))
    return stack


def slot(frame, conditioning=False):
    return MemorySlot(frame_index=frame, features=torch.zeros(4, 64),
                      positions=torch.zeros(4, 64), pointer=torch.arange(256).float() + frame * 256,
                      conditioning=conditioning)


def test_pointer_chunks_and_positions_share_object_order():
    stack = lightweight_stack()
    slots = [slot(0), slot(1)]
    got = stack._collect_pointers(slots, [2, 1])
    assert torch.equal(got, torch.arange(512).float().reshape(8, 64))


@pytest.mark.parametrize("forward", [True, False])
def test_long_sweeps_release_unattended_memory_without_changing_attention(forward):
    """Compare the bounded bank with an unpruned history, including retries."""
    bank = MemoryBank(7)
    history = MemoryBank(7)
    order = list(range(100)) if forward else list(reversed(range(100)))
    anchor = order[0]
    bank.store(slot(anchor, conditioning=True))
    history._slots[anchor] = slot(anchor, conditioning=True)
    for frame in order[1:]:
        for _ in range(2):
            actual, actual_offsets = bank.gather(frame, forward)
            expected, expected_offsets = history.gather(frame, forward)
            assert [item.frame_index for item in actual] == [item.frame_index for item in expected]
            assert actual_offsets == expected_offsets
            bank.store(slot(frame))
            history._slots[frame] = slot(frame)
        assert len(bank) <= 8  # anchor, current frame, six predecessors
        assert bank.get(anchor) is not None


def test_pointer_and_temporal_ablations_actually_remove_their_signals():
    stack = lightweight_stack()
    bank = MemoryBank(7)
    bank.store(slot(0, conditioning=True))
    bank.store(slot(1))
    bank.store(slot(9, conditioning=True))  # never let an ablation grant future access
    stack.object_pointers_enabled = False
    features, positions, count, indices = stack.build_attention_inputs(bank, 2, True)
    assert count == 0 and features.shape == (8, 64)
    assert indices == [0, 1]
    assert positions.eq(1).all()
    stack.temporal_position_encoding_enabled = False
    _, positions, _, indices = stack.build_attention_inputs(bank, 2, True)
    assert positions.eq(0).all() and 9 not in indices


def test_presence_ablation_disables_mask_gating_before_it_can_erase_logits():
    decoder = MaskDecoder(hidden_size=32, num_attention_heads=4, mlp_dim=64).eval()
    for parameter in decoder.pred_obj_score_head.parameters():
        parameter.data.zero_()
    # Force a definitely absent presence token.
    list(decoder.pred_obj_score_head.layers)[-1].bias.data.fill_(-10)
    inputs = dict(image_embeddings=torch.randn(1, 32, 2, 2),
                  image_positional_embeddings=torch.randn(1, 32, 2, 2),
                  sparse_prompt_embeddings=torch.randn(1, 1, 1, 32),
                  dense_prompt_embeddings=torch.randn(1, 32, 2, 2),
                  multimask_output=True,
                  high_resolution_features=[torch.randn(1, 4, 8, 8), torch.randn(1, 8, 4, 4)],
                  image_size=32)
    with torch.no_grad():
        absent = decoder(**inputs)
        decoder.presence_head_enabled = False
        disabled = decoder(**inputs)
    assert absent.low_res_masks.eq(-1024).all()
    assert disabled.object_score_logits.gt(0).all()
    assert torch.equal(disabled.low_res_masks, disabled.raw_low_res_masks)


def test_training_checkpoint_restores_named_stack_and_pretrained_frozen_encoder(tmp_path, monkeypatch):
    """The real trainer stores a wrapper and intentionally omits frozen weights."""
    from types import SimpleNamespace
    transformers = pytest.importorskip("transformers")
    import app.models.sam2_stack as stack_module
    from app.models.sam2_memory import MemoryAttentionTracker

    class TinyStack(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.stack_weight = torch.nn.Parameter(torch.zeros(1))
            self.vision_encoder = torch.nn.Linear(1, 1, bias=False)
            self.num_maskmem = 7
            self.mask_decoder = SimpleNamespace()

        def load_reference_weights(self, backbone):
            with torch.no_grad():
                self.vision_encoder.weight.fill_(7)
            return [], []

        def get_image_wide_positional_embeddings(self):
            return torch.zeros(1, 1, 1, 1)

    monkeypatch.setattr(stack_module, "MemoryStack", TinyStack)
    monkeypatch.setattr(transformers.Sam2VideoConfig, "from_pretrained", lambda _: SimpleNamespace())
    checkpoint = tmp_path / "training.pt"
    torch.save({"weights": {"kind": "state_dict", "state": {"stack_weight": torch.tensor([9.0])}}}, checkpoint)
    tracker = MemoryAttentionTracker()
    tracker.load(device="cpu", checkpoint=checkpoint)
    assert tracker.stack.stack_weight.item() == 9
    assert tracker.stack.vision_encoder.weight.item() == 7
    assert not tracker.stack.vision_encoder.weight.requires_grad
    bad = tmp_path / "wrong.pt"
    torch.save({"stack_wrong": torch.tensor([9.0])}, bad)
    with pytest.raises(ValueError, match="does not match"):
        tracker.load(device="cpu", checkpoint=bad)


def test_scratch_initialization_preserves_the_frozen_pretrained_encoder():
    stack = lightweight_stack()
    stack.vision_encoder = torch.nn.Linear(2, 2)
    stack.memory_projection = torch.nn.Linear(2, 2)
    stack.enable_occlusion_spatial_embedding = False
    stack.no_memory_embedding = torch.nn.Parameter(torch.ones(1, 1, 256))
    stack.no_memory_positional_encoding = torch.nn.Parameter(torch.ones(1, 1, 256))
    with torch.no_grad():
        stack.vision_encoder.weight.fill_(7)
        stack.vision_encoder.bias.fill_(3)
        stack.memory_projection.weight.fill_(9)
    stack.init_weights()
    assert stack.vision_encoder.weight.eq(7).all()
    assert stack.vision_encoder.bias.eq(3).all()
    assert not stack.vision_encoder.weight.requires_grad
    assert not stack.memory_projection.weight.eq(9).all()


def test_frozen_encoder_does_not_detach_trainable_decoder_skip_projections():
    from types import SimpleNamespace
    stack = lightweight_stack()

    class FrozenVision(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Parameter(torch.ones(1, 256, 4, 4))

        def forward(self, pixels):
            return SimpleNamespace(fpn_hidden_states=[self.embedding * 2] * 3,
                                   fpn_position_encoding=[self.embedding * 0] * 3)

    stack.vision_encoder = FrozenVision()
    stack.mask_decoder = torch.nn.Module()
    stack.mask_decoder.conv_s0 = torch.nn.Conv2d(256, 32, 1)
    stack.mask_decoder.conv_s1 = torch.nn.Conv2d(256, 64, 1)
    caches = stack.encode_frame(torch.zeros(1, 3, 8, 8))
    sum(level.sum() for level in caches.high_res).backward()
    assert stack.mask_decoder.conv_s0.weight.grad.abs().sum() > 0
    assert stack.mask_decoder.conv_s1.weight.grad.abs().sum() > 0
    assert stack.vision_encoder.embedding.grad is None
