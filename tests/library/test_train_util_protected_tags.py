import argparse
import random
import sys
import types
from importlib.machinery import ModuleSpec
from unittest.mock import patch

if "torchvision" not in sys.modules:
    class _NoOpTransform:
        def __call__(self, value):
            return value

    transforms_stub = types.SimpleNamespace(
        Compose=lambda items: items,
        ToTensor=lambda: _NoOpTransform(),
        Normalize=lambda *args, **kwargs: _NoOpTransform(),
    )
    torchvision_stub = types.ModuleType("torchvision")
    torchvision_stub.transforms = transforms_stub
    torchvision_stub.__spec__ = ModuleSpec("torchvision", loader=None)
    sys.modules["torchvision"] = torchvision_stub

if "cv2" not in sys.modules:
    cv2_stub = types.ModuleType("cv2")
    cv2_stub.__spec__ = ModuleSpec("cv2", loader=None)
    sys.modules["cv2"] = cv2_stub

from library import train_util
from library.config_util import BlueprintGenerator, ConfigSanitizer
from library.train_util import BaseDataset, BaseSubset


def create_subset(
    protected_tags_file=None,
    caption_tag_dropout_rate=1.0,
    shuffle_caption=True,
    keep_tokens_separator="",
    caption_mode="caption",
    mixed_weights=None,
):
    return BaseSubset(
        image_dir=None,
        alpha_mask=False,
        num_repeats=1,
        shuffle_caption=shuffle_caption,
        caption_separator=",",
        keep_tokens=0,
        keep_tokens_separator=keep_tokens_separator,
        caption_mode=caption_mode,
        mixed_weights=mixed_weights,
        protected_tags_file=protected_tags_file,
        secondary_separator=None,
        enable_wildcard=False,
        color_aug=False,
        flip_aug=False,
        face_crop_aug_range=None,
        random_crop=False,
        caption_dropout_rate=0.0,
        caption_dropout_every_n_epochs=0,
        caption_tag_dropout_rate=caption_tag_dropout_rate,
        caption_prefix=None,
        caption_suffix=None,
        token_warmup_min=1,
        token_warmup_step=0,
    )


def create_blueprint_args(**overrides):
    defaults = dict(
        train_batch_size=4,
        resolution=(1024, 1024),
        enable_bucket=True,
        min_bucket_reso=256,
        max_bucket_reso=2048,
        bucket_no_upscale=True,
        bucket_reso_steps=64,
        network_multiplier=1.0,
        caption_mode="mixed",
        mixed_weights={"tags": 25, "nl": 25, "tags_nl": 25, "nl_tags": 25},
        protected_tags_file="./protected_tags.txt",
        keep_tokens_separator="|||",
        secondary_separator=";;;",
        shuffle_caption=False,
        keep_tokens=0,
        caption_dropout_rate=0.05,
        caption_dropout_every_n_epochs=0,
        caption_tag_dropout_rate=0.1,
        caption_prefix=None,
        caption_suffix=None,
        token_warmup_min=1,
        token_warmup_step=0,
        color_aug=False,
        flip_aug=False,
        face_crop_aug_range=None,
        random_crop=False,
        caption_separator=",",
        enable_wildcard=False,
        validation_seed=None,
        validation_split=0.0,
        resize_interpolation=None,
        alpha_mask=False,
        debug_dataset=False,
        max_token_length=None,
        prior_loss_weight=1.0,
        dataset_repeats=1,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_protected_tags_are_not_dropped_but_are_still_shuffled(tmp_path):
    protected_tags_file = tmp_path / "protected_tags.txt"
    protected_tags_file.write_text("apple\nbanana\nloov\n", encoding="utf-8")

    dataset = BaseDataset(resolution=None, network_multiplier=1.0, debug_dataset=False)
    subset = create_subset(str(protected_tags_file))

    with patch("library.train_util.random.shuffle", side_effect=lambda items: items.reverse()):
        with patch("library.train_util.random.random", return_value=0.0):
            caption, caption_info = dataset.process_caption(subset, "apple, cat, banana, dog, loov", return_info=True)

    assert caption == "loov, banana, apple"
    assert caption_info["dropped_tags"] == ["dog", "cat"]


def test_protected_tags_match_case_insensitively_and_ignore_comments(tmp_path):
    protected_tags_file = tmp_path / "protected_tags.txt"
    protected_tags_file.write_text("# comment\nApple\n\nLOOV\n", encoding="utf-8")

    dataset = BaseDataset(resolution=None, network_multiplier=1.0, debug_dataset=False)
    subset = create_subset(str(protected_tags_file), shuffle_caption=False)

    with patch("library.train_util.random.random", return_value=0.0):
        caption = dataset.process_caption(subset, "apple, banana, loov")

    assert caption == "apple, loov"


def test_maybe_log_train_captions_logs_caption_and_dropped_tags(caplog):
    args = argparse.Namespace(log_captions_every_n_steps=100, log_captions_max_length=2)
    batch = {
        "captions": ["apple, banana, loov, kiwi"],
        "caption_infos": [
            {
                "processed_caption": "apple, banana, loov, kiwi",
                "dropped_tags": ["cat", "dog", "bird"],
                "is_caption_dropout": False,
                "image_key": "sample.png",
                "mixed_mode": "nl",
            }
        ],
    }

    with caplog.at_level("INFO"):
        train_util.maybe_log_train_captions(args, batch, 100)

    assert "captions at step 100" in caplog.text
    assert "[0] sample.png caption [nl]: apple, banana, ... (+2 more)" in caplog.text
    assert "[0] sample.png dropped tags: cat, dog, ... (+1 more)" in caplog.text


def test_log_protected_tags_epoch_start_logs_loaded_tags(caplog):
    subset = create_subset()
    subset.protected_tags_file = "./protected_tags.txt"
    subset.protected_tags = {"apple", "banana"}
    dataset_group = types.SimpleNamespace(datasets=[types.SimpleNamespace(subsets=[subset])])

    with caplog.at_level("INFO"):
        train_util.log_protected_tags_epoch_start(dataset_group, 1)

    assert "protected tags at epoch 1" in caplog.text
    assert "[Dataset 0 / Subset 0] 2 tags from ./protected_tags.txt" in caplog.text


def test_log_dataset_caption_config_mismatch_warns_when_dataset_config_overrides_args(caplog):
    args = argparse.Namespace(
        dataset_config="./dataset_config.toml",
        mixed_weights={"tags": 25, "nl": 25, "tags_nl": 25, "nl_tags": 25},
    )
    subset = create_subset(caption_mode="mixed")
    subset.mixed_weights = {"tags": 50.0, "nl": 10.0, "tags_nl": 20.0, "nl_tags": 20.0}
    dataset_group = types.SimpleNamespace(datasets=[types.SimpleNamespace(subsets=[subset])])

    with caplog.at_level("WARNING"):
        train_util.maybe_log_dataset_caption_config_mismatch(args, dataset_group)

    assert "does not match top-level args" in caplog.text
    assert "set it in the dataset_config as well" in caplog.text


def test_dataset_config_mixed_weights_override_top_level_args_only_when_explicitly_set():
    args = create_blueprint_args()
    sanitizer = ConfigSanitizer(True, True, False, True)

    blueprint_without_dataset_override = BlueprintGenerator(sanitizer).generate(
        {"general": {"caption_mode": "mixed"}, "datasets": [{"subsets": [{"image_dir": "./img"}]}]},
        args,
    )
    subset_without_override = blueprint_without_dataset_override.dataset_group.datasets[0].subsets[0].params
    assert subset_without_override.mixed_weights == {"tags": 25, "nl": 25, "tags_nl": 25, "nl_tags": 25}

    blueprint_with_dataset_override = BlueprintGenerator(sanitizer).generate(
        {
            "general": {
                "caption_mode": "mixed",
                "mixed_weights": {"tags": 50, "nl": 10, "tags_nl": 20, "nl_tags": 20},
            },
            "datasets": [{"subsets": [{"image_dir": "./img"}]}],
        },
        args,
    )
    subset_with_override = blueprint_with_dataset_override.dataset_group.datasets[0].subsets[0].params
    assert subset_with_override.mixed_weights == {"tags": 50, "nl": 10, "tags_nl": 20, "nl_tags": 20}


def test_mixed_caption_nl_tags_keeps_fixed_prefix_first():
    dataset = BaseDataset(resolution=None, network_multiplier=1.0, debug_dataset=False)
    subset = create_subset(
        caption_tag_dropout_rate=0.0,
        shuffle_caption=False,
        keep_tokens_separator="|||",
        caption_mode="mixed",
        mixed_weights={"nl_tags": 1},
    )

    caption, caption_info = dataset.process_caption(
        subset,
        {"tags": "a, b ||| c, d", "nl": "natural language, with commas"},
        return_info=True,
    )

    assert caption == "a, b ||| natural language, with commas, c, d"
    assert caption_info["mixed_mode"] == "nl_tags"


def test_mixed_caption_nl_is_not_affected_by_tag_dropout():
    dataset = BaseDataset(resolution=None, network_multiplier=1.0, debug_dataset=False)
    subset = create_subset(
        caption_tag_dropout_rate=1.0,
        shuffle_caption=False,
        keep_tokens_separator="|||",
        caption_mode="mixed",
        mixed_weights={"nl": 1},
    )

    caption, caption_info = dataset.process_caption(
        subset,
        {"tags": "a, b ||| c, d", "nl": "natural language, with commas"},
        return_info=True,
    )

    assert caption == "a, b ||| natural language, with commas"
    assert caption_info["dropped_tags"] == ["c", "d"]


def test_subset_has_backward_compatible_defaults_for_new_caption_attributes():
    subset = create_subset()

    del subset.caption_mode
    del subset.mixed_weights

    assert subset.caption_mode == "caption"
    assert subset.mixed_weights is None


def test_mixed_caption_equal_weights_can_select_all_modes():
    dataset = BaseDataset(resolution=None, network_multiplier=1.0, debug_dataset=False)
    subset = create_subset(
        caption_tag_dropout_rate=0.0,
        shuffle_caption=False,
        keep_tokens_separator="|||",
        caption_mode="mixed",
        mixed_weights={"tags": 25, "nl": 25, "tags_nl": 25, "nl_tags": 25},
    )

    random.seed(12345)
    selected_modes = set()
    for _ in range(200):
        _, caption_info = dataset.process_caption(
            subset,
            {"tags": "a, b ||| c, d", "nl": "natural language"},
            return_info=True,
        )
        selected_modes.add(caption_info["mixed_mode"])

    assert selected_modes == {"tags", "nl", "tags_nl", "nl_tags"}


def test_mixed_caption_uses_documented_default_weights_when_not_specified():
    dataset = BaseDataset(resolution=None, network_multiplier=1.0, debug_dataset=False)
    subset = create_subset(
        caption_tag_dropout_rate=0.0,
        shuffle_caption=False,
        keep_tokens_separator="|||",
        caption_mode="mixed",
        mixed_weights=None,
    )

    assert subset.mixed_weights == {"tags": 50.0, "nl": 10.0, "tags_nl": 20.0, "nl_tags": 20.0}

    random.seed(12345)
    selected_modes = set()
    for _ in range(200):
        _, caption_info = dataset.process_caption(
            subset,
            {"tags": "a, b ||| c, d", "nl": "natural language"},
            return_info=True,
        )
        selected_modes.add(caption_info["mixed_mode"])

    assert selected_modes == {"tags", "nl", "tags_nl", "nl_tags"}
