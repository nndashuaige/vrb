from pathlib import Path

from vrbreproduction.e10_cotracker_part_aware_contact_projection import E10Config, run_e10_experiment


if __name__ == "__main__":
    cfg = E10Config(
        num_subactions=1,
        output_root=Path("/root/workspace/vrb/vrbreproduction/outputs/100subaction-e10-smoke"),
    )
    run_e10_experiment(cfg)
