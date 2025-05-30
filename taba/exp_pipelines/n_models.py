from pathlib import Path
from typing import List

def get_all_ckpts(models_path: Path) -> List[int]:
    return sorted(
        [int(str(x).split("_")[-1].split(".")[0]) for x in models_path.glob("ema_*.pt")]
    )

def get_all_samples(samples_path: Path) -> List[int]:
    return sorted(
        [int(str(x).split("/")[-1].split(".")[0]) for x in samples_path.glob("*.pt")]
    )

# trainsteps_to_sample = sorted(
#     list(
#         set(
#             [0, 5, 10, 25, 50, 100]
#             + list(range(0, 10_308, 250))
#             + list(range(0, 522_500, 10_307))
#             + list(range(522_500, 1_130_613, 10_307))
#             + list(range(1_130_613, 1_500_614, 10_307))
#             + list(range(0, 1_130_613, 2_500))
#             + list(range(1_130_613, 1_500_614, 2_500))
#         )
#     )
# )

trainsteps_to_sample = sorted(
    list(
        set(
            [0, 5, 10, 25]
            + list(range(50, 401, 50))
            + list(range(0, 100 * 390 + 1, 390))
            + list(range(101 * 390, 429390, 5 * 390))
            + list(range(0, 700_001, 2_500))
        )
    )
)

if __name__ == "__main__":
    all_ckpts = get_all_ckpts(Path("res/ckpt_models/cifar10_32/seed_42").resolve())
    all_samples = get_all_samples(Path("experiments/trainsteps/cifar10_32/s42/samples").resolve())
    print("No models:")
    print([
        x for x in trainsteps_to_sample if x not in all_ckpts
    ])
    print("No samples:")
    print([
        x for x in trainsteps_to_sample if x not in all_samples
    ])
