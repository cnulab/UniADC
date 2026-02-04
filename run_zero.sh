kn=2

python create_experiments.py --normal_k_shot $kn --anomaly_k_shot 0 --category hazelnut --dataset mvtec
python zero_shot_gen.py --dataset mvtec --sample_file "zero_hazelnut_n${kn}.jsonl" --anomaly_type crack
python zero_shot_gen.py --dataset mvtec --sample_file "zero_hazelnut_n${kn}.jsonl" --anomaly_type cut
python zero_shot_gen.py --dataset mvtec --sample_file "zero_hazelnut_n${kn}.jsonl" --anomaly_type print
python zero_shot_gen.py --dataset mvtec --sample_file "zero_hazelnut_n${kn}.jsonl" --anomaly_type hole
python train_discriminator.py --dataset mvtec --sample_file "zero_hazelnut_n${kn}.jsonl" --usegen


python create_experiments.py --normal_k_shot $kn --anomaly_k_shot 0 --category metal_nut --dataset mvtec
python zero_shot_gen.py --dataset mvtec --sample_file "zero_metal_nut_n${kn}.jsonl" --anomaly_type color
python zero_shot_gen.py --dataset mvtec --sample_file "zero_metal_nut_n${kn}.jsonl" --anomaly_type scratch
python zero_shot_gen.py --dataset mvtec --sample_file "zero_metal_nut_n${kn}.jsonl" --anomaly_type flip
python zero_shot_gen.py --dataset mvtec --sample_file "zero_metal_nut_n${kn}.jsonl" --anomaly_type bent
python train_discriminator.py --dataset mvtec --sample_file "zero_metal_nut_n${kn}.jsonl" --usegen

python create_experiments.py --normal_k_shot $kn --anomaly_k_shot 0 --category wood --dataset mvtec
python zero_shot_gen.py --dataset mvtec --sample_file "zero_wood_n${kn}.jsonl" --anomaly_type hole
python zero_shot_gen.py --dataset mvtec --sample_file "zero_wood_n${kn}.jsonl" --anomaly_type color
python zero_shot_gen.py --dataset mvtec --sample_file "zero_wood_n${kn}.jsonl" --anomaly_type liquid
python zero_shot_gen.py --dataset mvtec --sample_file "zero_wood_n${kn}.jsonl" --anomaly_type scratch
python train_discriminator.py --dataset mvtec --sample_file "zero_wood_n${kn}.jsonl" --usegen


python create_experiments.py --normal_k_shot $kn --anomaly_k_shot 0 --category bottle --dataset mvtec
python zero_shot_gen.py --dataset mvtec --sample_file "zero_bottle_n${kn}.jsonl" --anomaly_type broken_large
python zero_shot_gen.py --dataset mvtec --sample_file "zero_bottle_n${kn}.jsonl" --anomaly_type broken_small
python zero_shot_gen.py --dataset mvtec --sample_file "zero_bottle_n${kn}.jsonl" --anomaly_type contamination
python train_discriminator.py --dataset mvtec --sample_file "zero_bottle_n${kn}.jsonl" --usegen


python create_experiments.py --normal_k_shot $kn --anomaly_k_shot 0 --category carpet --dataset mvtec
python zero_shot_gen.py --dataset mvtec --sample_file "zero_carpet_n${kn}.jsonl" --anomaly_type color
python zero_shot_gen.py --dataset mvtec --sample_file "zero_carpet_n${kn}.jsonl" --anomaly_type cut
python zero_shot_gen.py --dataset mvtec --sample_file "zero_carpet_n${kn}.jsonl" --anomaly_type metal_contamination
python zero_shot_gen.py --dataset mvtec --sample_file "zero_carpet_n${kn}.jsonl" --anomaly_type hole
python zero_shot_gen.py --dataset mvtec --sample_file "zero_carpet_n${kn}.jsonl" --anomaly_type thread
python train_discriminator.py --dataset mvtec --sample_file "zero_carpet_n${kn}.jsonl" --usegen

python create_experiments.py --normal_k_shot $kn --anomaly_k_shot 0 --category leather --dataset mvtec
python zero_shot_gen.py --dataset mvtec --sample_file "zero_leather_n${kn}.jsonl" --anomaly_type color
python zero_shot_gen.py --dataset mvtec --sample_file "zero_leather_n${kn}.jsonl" --anomaly_type cut
python zero_shot_gen.py --dataset mvtec --sample_file "zero_leather_n${kn}.jsonl" --anomaly_type fold
python zero_shot_gen.py --dataset mvtec --sample_file "zero_leather_n${kn}.jsonl" --anomaly_type glue
python zero_shot_gen.py --dataset mvtec --sample_file "zero_leather_n${kn}.jsonl" --anomaly_type poke
python train_discriminator.py --dataset mvtec --sample_file "zero_leather_n${kn}.jsonl" --usegen

python create_experiments.py --normal_k_shot $kn --anomaly_k_shot 0 --category tile --dataset mvtec
python zero_shot_gen.py --dataset mvtec --sample_file "zero_tile_n${kn}.jsonl" --anomaly_type oil
python zero_shot_gen.py --dataset mvtec --sample_file "zero_tile_n${kn}.jsonl" --anomaly_type glue_strip
python zero_shot_gen.py --dataset mvtec --sample_file "zero_tile_n${kn}.jsonl" --anomaly_type crack
python zero_shot_gen.py --dataset mvtec --sample_file "zero_tile_n${kn}.jsonl" --anomaly_type rough
python zero_shot_gen.py --dataset mvtec --sample_file "zero_tile_n${kn}.jsonl" --anomaly_type gray_stroke
python train_discriminator.py --dataset mvtec --sample_file "zero_tile_n${kn}.jsonl" --usegen


python create_experiments.py --normal_k_shot $kn --anomaly_k_shot 0 --category toothbrush --dataset mvtec
python zero_shot_gen.py --dataset mvtec --sample_file "zero_toothbrush_n${kn}.jsonl" --anomaly_type defective
python train_discriminator.py --dataset mvtec --sample_file "zero_toothbrush_n${kn}.jsonl" --usegen


python create_experiments.py --normal_k_shot $kn --anomaly_k_shot 0 --category cable --dataset mvtec
python zero_shot_gen.py --dataset mvtec --sample_file "zero_cable_n${kn}.jsonl" --anomaly_type bent_wire
python zero_shot_gen.py --dataset mvtec --sample_file "zero_cable_n${kn}.jsonl" --anomaly_type cut_inner_insulation
python zero_shot_gen.py --dataset mvtec --sample_file "zero_cable_n${kn}.jsonl" --anomaly_type cable_swap
python zero_shot_gen.py --dataset mvtec --sample_file "zero_cable_n${kn}.jsonl" --anomaly_type poke_insulation
python zero_shot_gen.py --dataset mvtec --sample_file "zero_cable_n${kn}.jsonl" --anomaly_type cut_outer_insulation
python zero_shot_gen.py --dataset mvtec --sample_file "zero_cable_n${kn}.jsonl" --anomaly_type missing_cable
python zero_shot_gen.py --dataset mvtec --sample_file "zero_cable_n${kn}.jsonl" --anomaly_type missing_wire
python train_discriminator.py --dataset mvtec --sample_file "zero_cable_n${kn}.jsonl" --usegen


python create_experiments.py --normal_k_shot $kn --anomaly_k_shot 0 --category capsule --dataset mvtec
python zero_shot_gen.py --dataset mvtec --sample_file "zero_capsule_n${kn}.jsonl" --anomaly_type crack
python zero_shot_gen.py --dataset mvtec --sample_file "zero_capsule_n${kn}.jsonl" --anomaly_type faulty_imprint
python zero_shot_gen.py --dataset mvtec --sample_file "zero_capsule_n${kn}.jsonl" --anomaly_type scratch
python zero_shot_gen.py --dataset mvtec --sample_file "zero_capsule_n${kn}.jsonl" --anomaly_type poke
python zero_shot_gen.py --dataset mvtec --sample_file "zero_capsule_n${kn}.jsonl" --anomaly_type squeeze
python train_discriminator.py --dataset mvtec --sample_file "zero_capsule_n${kn}.jsonl" --usegen


python create_experiments.py --normal_k_shot $kn --anomaly_k_shot 0 --category grid --dataset mvtec
python zero_shot_gen.py --dataset mvtec --sample_file "zero_grid_n${kn}.jsonl" --anomaly_type glue
python zero_shot_gen.py --dataset mvtec --sample_file "zero_grid_n${kn}.jsonl" --anomaly_type metal_contamination
python zero_shot_gen.py --dataset mvtec --sample_file "zero_grid_n${kn}.jsonl" --anomaly_type broken
python zero_shot_gen.py --dataset mvtec --sample_file "zero_grid_n${kn}.jsonl" --anomaly_type bent
python zero_shot_gen.py --dataset mvtec --sample_file "zero_grid_n${kn}.jsonl" --anomaly_type thread
python train_discriminator.py --dataset mvtec --sample_file "zero_grid_n${kn}.jsonl" --usegen


python create_experiments.py --normal_k_shot $kn --anomaly_k_shot 0 --category pill --dataset mvtec
python zero_shot_gen.py --dataset mvtec --sample_file "zero_pill_n${kn}.jsonl" --anomaly_type color
python zero_shot_gen.py --dataset mvtec --sample_file "zero_pill_n${kn}.jsonl" --anomaly_type contamination
python zero_shot_gen.py --dataset mvtec --sample_file "zero_pill_n${kn}.jsonl" --anomaly_type scratch
python zero_shot_gen.py --dataset mvtec --sample_file "zero_pill_n${kn}.jsonl" --anomaly_type crack
python zero_shot_gen.py --dataset mvtec --sample_file "zero_pill_n${kn}.jsonl" --anomaly_type pill_type
python zero_shot_gen.py --dataset mvtec --sample_file "zero_pill_n${kn}.jsonl" --anomaly_type faulty_imprint
python train_discriminator.py --dataset mvtec --sample_file "zero_pill_n${kn}.jsonl" --usegen


python create_experiments.py --normal_k_shot $kn --anomaly_k_shot 0 --category screw --dataset mvtec
python zero_shot_gen.py --dataset mvtec --sample_file "zero_screw_n${kn}.jsonl" --anomaly_type manipulated_front
python zero_shot_gen.py --dataset mvtec --sample_file "zero_screw_n${kn}.jsonl" --anomaly_type scratch_head
python zero_shot_gen.py --dataset mvtec --sample_file "zero_screw_n${kn}.jsonl" --anomaly_type scratch_neck
python zero_shot_gen.py --dataset mvtec --sample_file "zero_screw_n${kn}.jsonl" --anomaly_type thread_top
python zero_shot_gen.py --dataset mvtec --sample_file "zero_screw_n${kn}.jsonl" --anomaly_type thread_side
python train_discriminator.py --dataset mvtec --sample_file "zero_screw_n${kn}.jsonl" --usegen


python create_experiments.py --normal_k_shot $kn --anomaly_k_shot 0 --category transistor --dataset mvtec
python zero_shot_gen.py --dataset mvtec --sample_file "zero_transistor_n${kn}.jsonl" --anomaly_type damaged_case
python zero_shot_gen.py --dataset mvtec --sample_file "zero_transistor_n${kn}.jsonl" --anomaly_type bent_lead
python zero_shot_gen.py --dataset mvtec --sample_file "zero_transistor_n${kn}.jsonl" --anomaly_type cut_lead
python zero_shot_gen.py --dataset mvtec --sample_file "zero_transistor_n${kn}.jsonl" --anomaly_type misplaced
python train_discriminator.py --dataset mvtec --sample_file "zero_transistor_n${kn}.jsonl" --usegen


python create_experiments.py --normal_k_shot $kn --anomaly_k_shot 0 --category zipper --dataset mvtec
python zero_shot_gen.py --dataset mvtec --sample_file "zero_zipper_n${kn}.jsonl" --anomaly_type fabric_border
python zero_shot_gen.py --dataset mvtec --sample_file "zero_zipper_n${kn}.jsonl" --anomaly_type fabric_interior
python zero_shot_gen.py --dataset mvtec --sample_file "zero_zipper_n${kn}.jsonl" --anomaly_type split_teeth
python zero_shot_gen.py --dataset mvtec --sample_file "zero_zipper_n${kn}.jsonl" --anomaly_type broken_teeth
python zero_shot_gen.py --dataset mvtec --sample_file "zero_zipper_n${kn}.jsonl" --anomaly_type rough
python zero_shot_gen.py --dataset mvtec --sample_file "zero_zipper_n${kn}.jsonl" --anomaly_type squeezed_teeth
python train_discriminator.py --dataset mvtec --sample_file "zero_zipper_n${kn}.jsonl" --usegen


python create_experiments.py --dataset wfdd  --normal_k_shot $kn --anomaly_k_shot 0 --category grey_cloth
python zero_shot_gen.py --sample_file "zero_grey_cloth_n${kn}.jsonl" --dataset wfdd --anomaly_type contaminated
python zero_shot_gen.py --sample_file "zero_grey_cloth_n${kn}.jsonl" --dataset wfdd --anomaly_type flecked
python zero_shot_gen.py --sample_file "zero_grey_cloth_n${kn}.jsonl" --dataset wfdd --anomaly_type line
python zero_shot_gen.py --sample_file "zero_grey_cloth_n${kn}.jsonl" --dataset wfdd --anomaly_type string
python train_discriminator.py --sample_file "zero_grey_cloth_n${kn}.jsonl" --dataset wfdd --usegen


python create_experiments.py --dataset wfdd  --normal_k_shot $kn --anomaly_k_shot 0 --category grid_cloth
python zero_shot_gen.py --sample_file "zero_grid_cloth_n${kn}.jsonl" --dataset wfdd --anomaly_type string
python zero_shot_gen.py --sample_file "zero_grid_cloth_n${kn}.jsonl" --dataset wfdd --anomaly_type flecked
python zero_shot_gen.py --sample_file "zero_grid_cloth_n${kn}.jsonl" --dataset wfdd --anomaly_type fold
python train_discriminator.py --sample_file "zero_grid_cloth_n${kn}.jsonl" --dataset wfdd --usegen


python create_experiments.py --dataset wfdd  --normal_k_shot $kn --anomaly_k_shot 0 --category pink_flower
python zero_shot_gen.py --sample_file "zero_pink_flower_n${kn}.jsonl" --dataset wfdd --anomaly_type hole
python zero_shot_gen.py --sample_file "zero_pink_flower_n${kn}.jsonl" --dataset wfdd --anomaly_type stain
python zero_shot_gen.py --sample_file "zero_pink_flower_n${kn}.jsonl" --dataset wfdd --anomaly_type tear
python train_discriminator.py --sample_file "zero_pink_flower_n${kn}.jsonl" --dataset wfdd --usegen


python create_experiments.py --dataset wfdd  --normal_k_shot $kn --anomaly_k_shot 0 --category yellow_cloth
python zero_shot_gen.py --sample_file "zero_yellow_cloth_n${kn}.jsonl" --dataset wfdd --anomaly_type fold
python zero_shot_gen.py --sample_file "zero_yellow_cloth_n${kn}.jsonl" --dataset wfdd --anomaly_type stain
python zero_shot_gen.py --sample_file "zero_yellow_cloth_n${kn}.jsonl" --dataset wfdd --anomaly_type string
python train_discriminator.py --sample_file "zero_yellow_cloth_n${kn}.jsonl" --dataset wfdd --usegen
