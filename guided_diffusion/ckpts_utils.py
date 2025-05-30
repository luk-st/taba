ADDITIONAL_SAVESTEPS_CIFAR = sorted(
    list(
        set(
            [5, 10, 25]
            + list(range(50, 401, 50))
            + list(range(0, 100 * 390 + 1, 390))
            + list(range(101 * 390, 500_001, 5 * 390))
        )
    )
)
ADDITIONAL_SAVESTEPS_IMAGENET = sorted(
    list(set([5, 10, 25, 50, 100] + list(range(0, 10_308, 250)) + list(range(0, 1_500_001, 10_307))))
)
