import torch
from torch import nn

from preprocessing.data import stratified_split_indices
from preprocessing.model import PaperPreprocessor, VideoTransformerPreprocessor
from preprocessing.standard_codec import ParallelStandardVideoCodec, StandardCodecProxy


def test_preprocessor_shape_range_and_identity_initialization():
    model = PaperPreprocessor(temporal_frames=4)
    clip = torch.rand(2, 6, 3, 32, 40)
    output = model(clip)
    assert output.shape == clip.shape
    assert output.min() >= 0
    assert output.max() <= 1
    torch.testing.assert_close(output, clip)


def test_preprocessor_backpropagates():
    model = PaperPreprocessor(temporal_frames=3)
    clip = torch.rand(1, 4, 3, 16, 16)
    model(clip).mean().backward()
    assert model.to_rgb.weight.grad is not None


def test_vit_preprocessor_handles_non_patch_multiple_and_starts_as_identity():
    model = VideoTransformerPreprocessor(
        patch_size=4, embed_dim=16, depth=2, num_heads=4
    )
    clip = torch.rand(1, 3, 3, 17, 19)
    output = model(clip)
    assert output.shape == clip.shape
    torch.testing.assert_close(output, clip)


def test_vit_preprocessor_backpropagates_to_residual_head():
    model = VideoTransformerPreprocessor(
        patch_size=4, embed_dim=16, depth=2, num_heads=4
    )
    clip = torch.rand(1, 2, 3, 16, 16)
    model(clip).mean().backward()
    assert model.to_rgb.weight.grad is not None


def test_standard_codec_proxy_preserves_clip_shape_and_has_input_gradient():
    proxy = StandardCodecProxy(hidden_channels=8, latent_channels=12)
    clip = torch.rand(1, 3, 3, 17, 19, requires_grad=True)
    reconstruction, bpp = proxy(clip, qp=35)
    assert reconstruction.shape == clip.shape
    assert bpp.shape == (1,)
    (reconstruction.mean() + bpp.mean()).backward()
    assert clip.grad is not None
    assert torch.isfinite(clip.grad).all()


class _FakeStandardCodec(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qp = 35

    def set_qp(self, qp: int) -> None:
        self.qp = qp

    def forward(self, clip: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = clip.shape[0]
        return torch.full_like(clip, 0.375), torch.full((batch,), 1.25, device=clip.device)


class _FakeProxy(nn.Module):
    def forward(
        self, clip: torch.Tensor, qp: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del qp
        return clip * 2.0, clip.mean(dim=(1, 2, 3, 4))


def test_parallel_codec_uses_real_forward_values_and_proxy_backward():
    clip = torch.rand(1, 2, 3, 8, 8, requires_grad=True)
    codec = ParallelStandardVideoCodec(_FakeStandardCodec(), _FakeProxy()).train()
    reconstruction, bpp = codec(clip)
    torch.testing.assert_close(reconstruction, torch.full_like(clip, 0.375))
    torch.testing.assert_close(bpp, torch.tensor([1.25]))
    (reconstruction.mean() + bpp.mean()).backward()
    assert clip.grad is not None
    assert torch.count_nonzero(clip.grad) == clip.numel()


def test_stratified_split_has_no_overlap_and_preserves_classes():
    samples = [(None, label) for label in range(3) for _ in range(5)]
    train, val = stratified_split_indices(samples, val_ratio=0.2, seed=42)
    assert not set(train) & set(val)
    assert len(train) == 12
    assert len(val) == 3
    assert {samples[index][1] for index in train} == {0, 1, 2}
    assert {samples[index][1] for index in val} == {0, 1, 2}
