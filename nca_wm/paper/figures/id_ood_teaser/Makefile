.PHONY: all frames clean

# Final artifact: teaser.pdf, picked up by sections/intro.tex.
all: teaser.pdf

teaser.pdf: teaser.tex frames
	pdflatex -interaction=nonstopmode teaser.tex >/dev/null

# Regenerate the engine + model rollouts that teaser.tex includegraphics's.
# Uses the Train-199 cond checkpoint for OOD predictions; runs on CPU
# since the GPUs are usually busy with training.
RUN_DIR ?= ../../../logs/multi_scaling_gallery_v4_cond_match_s0
ID_GAMES ?= nekopuzzle,sokoban_basic,Modality
OOD_GAMES ?= in_the_way_by_giovanni_mota,my_nana_ate_my_kid_brother_by_ruth_williams,Hamiltwo

frames:
	cd ../../../.. && .venv/bin/python3 -m nca_wm.scripts.dump_engine_frames \
	    --games $(ID_GAMES) \
	    --out_dir nca_wm/paper/figures/id_ood_teaser/rollouts
	cd ../../../.. && JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
	    .venv/bin/python3 -m nca_wm.scripts.dump_ood_rollout_frames \
	    --run_dir $(RUN_DIR:../../../=)nca_wm/logs/multi_scaling_gallery_v4_cond_match_s0 \
	    --games $(OOD_GAMES) --n_steps 4 \
	    --out_dir nca_wm/paper/figures/id_ood_teaser/rollouts

clean:
	rm -f teaser.aux teaser.log teaser.pdf teaser.png
