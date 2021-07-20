DEFAULT_KWARGS = dict(
    batch_size=128,
    test_set_batch_size=64,
    num_epochs=10,
    momentum=0.9,
    lr=0.1,
    target_accuracy=0.0,
)

MODEL_SPECIFIC_DEFAULTS = {
    # Override some of the args in DEFAULT_KWARGS, or add them to the dict
    # if they don't exist.
    "resnet50": dict(
        DEFAULT_KWARGS,
        **{
            "lr": 0.5,
            "lr_scheduler_divide_every_n_epochs": 20,
            "lr_scheduler_divisor": 5,
            "lr_scheduler_type": "WarmupAndExponentialDecayScheduler",
        }
    ),
    "densenet121": dict(
        DEFAULT_KWARGS,
        **{
            "batch_size": 96,
            "lr": 0.5,
            "lr_scheduler_divide_every_n_epochs": 20,
            "lr_scheduler_divisor": 5,
            "lr_scheduler_type": "WarmupAndExponentialDecayScheduler",
        }
    ),
}
