kn=2
ka=1


python create_experiments.py  --dataset mvtec --normal_k_shot $kn --anomaly_k_shot $ka --category hazelnut
python few_shot_gen.py --dataset mvtec --sample_file "few_hazelnut_n${kn}a${ka}.jsonl" --anomaly_type crack
python few_shot_gen.py --dataset mvtec --sample_file "few_hazelnut_n${kn}a${ka}.jsonl" --anomaly_type cut
python few_shot_gen.py --dataset mvtec --sample_file "few_hazelnut_n${kn}a${ka}.jsonl" --anomaly_type print
python few_shot_gen.py --dataset mvtec --sample_file "few_hazelnut_n${kn}a${ka}.jsonl" --anomaly_type hole
python train_discriminator.py --dataset mvtec --sample_file "few_hazelnut_n${kn}a${ka}.jsonl" --usegen

python create_experiments.py  --dataset mvtec --normal_k_shot $kn --anomaly_k_shot $ka --category metal_nut
python few_shot_gen.py --dataset mvtec --sample_file "few_metal_nut_n${kn}a${ka}.jsonl" --anomaly_type color
python few_shot_gen.py --dataset mvtec --sample_file "few_metal_nut_n${kn}a${ka}.jsonl" --anomaly_type scratch
python few_shot_gen.py --dataset mvtec --sample_file "few_metal_nut_n${kn}a${ka}.jsonl" --anomaly_type flip
python few_shot_gen.py --dataset mvtec --sample_file "few_metal_nut_n${kn}a${ka}.jsonl" --anomaly_type bent
python train_discriminator.py --dataset mvtec --sample_file "few_metal_nut_n${kn}a${ka}.jsonl" --usegen

python create_experiments.py  --dataset mvtec --normal_k_shot $kn --anomaly_k_shot $ka --category wood
python few_shot_gen.py --dataset mvtec --sample_file "few_wood_n${kn}a${ka}.jsonl" --anomaly_type hole
python few_shot_gen.py --dataset mvtec --sample_file "few_wood_n${kn}a${ka}.jsonl" --anomaly_type color
python few_shot_gen.py --dataset mvtec --sample_file "few_wood_n${kn}a${ka}.jsonl" --anomaly_type liquid
python few_shot_gen.py --dataset mvtec --sample_file "few_wood_n${kn}a${ka}.jsonl" --anomaly_type scratch
python train_discriminator.py --dataset mvtec --sample_file "few_wood_n${kn}a${ka}.jsonl" --usegen


python create_experiments.py --dataset mvtec --normal_k_shot $kn --anomaly_k_shot $ka --category bottle
python few_shot_gen.py --dataset mvtec --sample_file "few_bottle_n${kn}a${ka}.jsonl" --anomaly_type broken_large
python few_shot_gen.py --dataset mvtec --sample_file "few_bottle_n${kn}a${ka}.jsonl" --anomaly_type broken_small
python few_shot_gen.py --dataset mvtec --sample_file "few_bottle_n${kn}a${ka}.jsonl" --anomaly_type contamination
python train_discriminator.py --dataset mvtec --sample_file "few_bottle_n${kn}a${ka}.jsonl" --usegen


python create_experiments.py  --dataset mvtec --normal_k_shot $kn --anomaly_k_shot $ka --category carpet
python few_shot_gen.py --dataset mvtec --sample_file "few_carpet_n${kn}a${ka}.jsonl" --anomaly_type color
python few_shot_gen.py --dataset mvtec --sample_file "few_carpet_n${kn}a${ka}.jsonl" --anomaly_type cut
python few_shot_gen.py --dataset mvtec --sample_file "few_carpet_n${kn}a${ka}.jsonl" --anomaly_type metal_contamination
python few_shot_gen.py --dataset mvtec --sample_file "few_carpet_n${kn}a${ka}.jsonl" --anomaly_type hole
python few_shot_gen.py --dataset mvtec --sample_file "few_carpet_n${kn}a${ka}.jsonl" --anomaly_type thread
python train_discriminator.py --dataset mvtec --sample_file "few_carpet_n${kn}a${ka}.jsonl" --usegen


python create_experiments.py  --dataset mvtec --normal_k_shot $kn --anomaly_k_shot $ka --category leather
python few_shot_gen.py --dataset mvtec --sample_file "few_leather_n${kn}a${ka}.jsonl" --anomaly_type color
python few_shot_gen.py --dataset mvtec --sample_file "few_leather_n${kn}a${ka}.jsonl" --anomaly_type cut
python few_shot_gen.py --dataset mvtec --sample_file "few_leather_n${kn}a${ka}.jsonl" --anomaly_type fold
python few_shot_gen.py --dataset mvtec --sample_file "few_leather_n${kn}a${ka}.jsonl" --anomaly_type glue
python few_shot_gen.py --dataset mvtec --sample_file "few_leather_n${kn}a${ka}.jsonl" --anomaly_type poke
python train_discriminator.py --dataset mvtec --sample_file "few_leather_n${kn}a${ka}.jsonl" --usegen


python create_experiments.py  --dataset mvtec --normal_k_shot $kn --anomaly_k_shot $ka --category tile
python few_shot_gen.py --dataset mvtec --sample_file "few_tile_n${kn}a${ka}.jsonl" --anomaly_type oil
python few_shot_gen.py --dataset mvtec --sample_file "few_tile_n${kn}a${ka}.jsonl" --anomaly_type glue_strip
python few_shot_gen.py --dataset mvtec --sample_file "few_tile_n${kn}a${ka}.jsonl" --anomaly_type crack
python few_shot_gen.py --dataset mvtec --sample_file "few_tile_n${kn}a${ka}.jsonl" --anomaly_type rough
python few_shot_gen.py --dataset mvtec --sample_file "few_tile_n${kn}a${ka}.jsonl" --anomaly_type gray_stroke
python train_discriminator.py --dataset mvtec --sample_file "few_tile_n${kn}a${ka}.jsonl" --usegen


python create_experiments.py  --dataset mvtec --normal_k_shot $kn --anomaly_k_shot $ka --category toothbrush
python few_shot_gen.py --dataset mvtec --sample_file "few_toothbrush_n${kn}a${ka}.jsonl" --anomaly_type defective
python train_discriminator.py --dataset mvtec --sample_file "few_toothbrush_n${kn}a${ka}.jsonl" --usegen


python create_experiments.py --dataset mvtec --normal_k_shot $kn --anomaly_k_shot $ka --category cable
python few_shot_gen.py --dataset mvtec --sample_file "few_cable_n${kn}a${ka}.jsonl" --anomaly_type bent_wire
python few_shot_gen.py --dataset mvtec --sample_file "few_cable_n${kn}a${ka}.jsonl" --anomaly_type cut_inner_insulation
python few_shot_gen.py --dataset mvtec --sample_file "few_cable_n${kn}a${ka}.jsonl" --anomaly_type cable_swap
python few_shot_gen.py --dataset mvtec --sample_file "few_cable_n${kn}a${ka}.jsonl" --anomaly_type poke_insulation
python few_shot_gen.py --dataset mvtec --sample_file "few_cable_n${kn}a${ka}.jsonl" --anomaly_type cut_outer_insulation
python few_shot_gen.py --dataset mvtec --sample_file "few_cable_n${kn}a${ka}.jsonl" --anomaly_type missing_cable
python few_shot_gen.py --dataset mvtec --sample_file "few_cable_n${kn}a${ka}.jsonl" --anomaly_type missing_wire
python train_discriminator.py --dataset mvtec --sample_file "few_cable_n${kn}a${ka}.jsonl" --usegen


python create_experiments.py  --dataset mvtec --normal_k_shot $kn --anomaly_k_shot $ka --category capsule
python few_shot_gen.py --dataset mvtec --sample_file "few_capsule_n${kn}a${ka}.jsonl" --anomaly_type crack
python few_shot_gen.py --dataset mvtec --sample_file "few_capsule_n${kn}a${ka}.jsonl" --anomaly_type faulty_imprint
python few_shot_gen.py --dataset mvtec --sample_file "few_capsule_n${kn}a${ka}.jsonl" --anomaly_type scratch
python few_shot_gen.py --dataset mvtec --sample_file "few_capsule_n${kn}a${ka}.jsonl" --anomaly_type poke
python few_shot_gen.py --dataset mvtec --sample_file "few_capsule_n${kn}a${ka}.jsonl" --anomaly_type squeeze
python train_discriminator.py --dataset mvtec --sample_file "few_capsule_n${kn}a${ka}.jsonl" --usegen


python create_experiments.py  --dataset mvtec --normal_k_shot $kn --anomaly_k_shot $ka --category grid
python few_shot_gen.py --dataset mvtec --sample_file "few_grid_n${kn}a${ka}.jsonl" --anomaly_type glue
python few_shot_gen.py --dataset mvtec --sample_file "few_grid_n${kn}a${ka}.jsonl" --anomaly_type metal_contamination
python few_shot_gen.py --dataset mvtec --sample_file "few_grid_n${kn}a${ka}.jsonl" --anomaly_type broken
python few_shot_gen.py --dataset mvtec --sample_file "few_grid_n${kn}a${ka}.jsonl" --anomaly_type bent
python few_shot_gen.py --dataset mvtec --sample_file "few_grid_n${kn}a${ka}.jsonl" --anomaly_type thread
python train_discriminator.py --dataset mvtec --sample_file "few_grid_n${kn}a${ka}.jsonl" --usegen


python create_experiments.py  --dataset mvtec --normal_k_shot $kn --anomaly_k_shot $ka --category pill
python few_shot_gen.py --dataset mvtec --sample_file "few_pill_n${kn}a${ka}.jsonl" --anomaly_type color
python few_shot_gen.py --dataset mvtec --sample_file "few_pill_n${kn}a${ka}.jsonl" --anomaly_type contamination
python few_shot_gen.py --dataset mvtec --sample_file "few_pill_n${kn}a${ka}.jsonl" --anomaly_type scratch
python few_shot_gen.py --dataset mvtec --sample_file "few_pill_n${kn}a${ka}.jsonl" --anomaly_type crack
python few_shot_gen.py --dataset mvtec --sample_file "few_pill_n${kn}a${ka}.jsonl" --anomaly_type pill_type
python few_shot_gen.py --dataset mvtec --sample_file "few_pill_n${kn}a${ka}.jsonl" --anomaly_type faulty_imprint
python train_discriminator.py --dataset mvtec --sample_file "few_pill_n${kn}a${ka}.jsonl" --usegen


python create_experiments.py  --dataset mvtec --normal_k_shot $kn --anomaly_k_shot $ka --category screw
python few_shot_gen.py --dataset mvtec --sample_file "few_screw_n${kn}a${ka}.jsonl" --anomaly_type manipulated_front
python few_shot_gen.py --dataset mvtec --sample_file "few_screw_n${kn}a${ka}.jsonl" --anomaly_type scratch_head
python few_shot_gen.py --dataset mvtec --sample_file "few_screw_n${kn}a${ka}.jsonl" --anomaly_type scratch_neck
python few_shot_gen.py --dataset mvtec --sample_file "few_screw_n${kn}a${ka}.jsonl" --anomaly_type thread_top
python few_shot_gen.py --dataset mvtec --sample_file "few_screw_n${kn}a${ka}.jsonl" --anomaly_type thread_side
python train_discriminator.py --dataset mvtec --sample_file "few_screw_n${kn}a${ka}.jsonl" --usegen


python create_experiments.py  --dataset mvtec --normal_k_shot $kn --anomaly_k_shot $ka --category transistor
python few_shot_gen.py --dataset mvtec --sample_file "few_transistor_n${kn}a${ka}.jsonl" --anomaly_type damaged_case
python few_shot_gen.py --dataset mvtec --sample_file "few_transistor_n${kn}a${ka}.jsonl" --anomaly_type bent_lead
python few_shot_gen.py --dataset mvtec --sample_file "few_transistor_n${kn}a${ka}.jsonl" --anomaly_type cut_lead
python few_shot_gen.py --dataset mvtec --sample_file "few_transistor_n${kn}a${ka}.jsonl" --anomaly_type misplaced
python train_discriminator.py --dataset mvtec --sample_file "few_transistor_n${kn}a${ka}.jsonl" --usegen


python create_experiments.py  --dataset mvtec --normal_k_shot $kn --anomaly_k_shot $ka --category zipper
python few_shot_gen.py --dataset mvtec --sample_file "few_zipper_n${kn}a${ka}.jsonl" --anomaly_type fabric_border
python few_shot_gen.py --dataset mvtec --sample_file "few_zipper_n${kn}a${ka}.jsonl" --anomaly_type fabric_interior
python few_shot_gen.py --dataset mvtec --sample_file "few_zipper_n${kn}a${ka}.jsonl" --anomaly_type split_teeth
python few_shot_gen.py --dataset mvtec --sample_file "few_zipper_n${kn}a${ka}.jsonl" --anomaly_type broken_teeth
python few_shot_gen.py --dataset mvtec --sample_file "few_zipper_n${kn}a${ka}.jsonl" --anomaly_type rough
python few_shot_gen.py --dataset mvtec --sample_file "few_zipper_n${kn}a${ka}.jsonl" --anomaly_type squeezed_teeth
python train_discriminator.py --dataset mvtec --sample_file "few_zipper_n${kn}a${ka}.jsonl" --usegen


python create_experiments.py --dataset wfdd  --normal_k_shot $kn --anomaly_k_shot $ka --category grey_cloth
python few_shot_gen.py --sample_file "few_grey_cloth_n${kn}a${ka}.jsonl" --dataset wfdd --anomaly_type contaminated
python few_shot_gen.py --sample_file "few_grey_cloth_n${kn}a${ka}.jsonl" --dataset wfdd --anomaly_type flecked
python few_shot_gen.py --sample_file "few_grey_cloth_n${kn}a${ka}.jsonl" --dataset wfdd --anomaly_type line
python few_shot_gen.py --sample_file "few_grey_cloth_n${kn}a${ka}.jsonl" --dataset wfdd --anomaly_type string
python train_discriminator.py --sample_file "few_grey_cloth_n${kn}a${ka}.jsonl" --dataset wfdd --usegen


python create_experiments.py --dataset wfdd  --normal_k_shot $kn --anomaly_k_shot $ka --category grid_cloth
python few_shot_gen.py --sample_file "few_grid_cloth_n${kn}a${ka}.jsonl" --dataset wfdd --anomaly_type string
python few_shot_gen.py --sample_file "few_grid_cloth_n${kn}a${ka}.jsonl" --dataset wfdd --anomaly_type flecked
python few_shot_gen.py --sample_file "few_grid_cloth_n${kn}a${ka}.jsonl" --dataset wfdd --anomaly_type fold
python train_discriminator.py --sample_file "few_grid_cloth_n${kn}a${ka}.jsonl" --dataset wfdd --usegen


python create_experiments.py --dataset wfdd  --normal_k_shot $kn --anomaly_k_shot $ka --category pink_flower
python few_shot_gen.py --sample_file "few_pink_flower_n${kn}a${ka}.jsonl" --dataset wfdd --anomaly_type hole
python few_shot_gen.py --sample_file "few_pink_flower_n${kn}a${ka}.jsonl" --dataset wfdd --anomaly_type stain
python few_shot_gen.py --sample_file "few_pink_flower_n${kn}a${ka}.jsonl" --dataset wfdd --anomaly_type tear
python train_discriminator.py --sample_file "few_pink_flower_n${kn}a${ka}.jsonl" --dataset wfdd --usegen


python create_experiments.py --dataset wfdd  --normal_k_shot $kn --anomaly_k_shot $ka --category yellow_cloth
python few_shot_gen.py --sample_file "few_yellow_cloth_n${kn}a${ka}.jsonl" --dataset wfdd --anomaly_type fold
python few_shot_gen.py --sample_file "few_yellow_cloth_n${kn}a${ka}.jsonl" --dataset wfdd --anomaly_type stain
python few_shot_gen.py --sample_file "few_yellow_cloth_n${kn}a${ka}.jsonl" --dataset wfdd --anomaly_type string
python train_discriminator.py --sample_file "few_yellow_cloth_n${kn}a${ka}.jsonl" --dataset wfdd --usegen