"""Curated multi-game ``scaling_gallery_v{1..5}`` presets for NCA world-model training.

These are hand-curated, nested game lists (each version a near-superset of the
previous) that were used for the dataset-size *scaling* experiments. They were
filtered manually by complexity (rules / objects / level area) and deduplicated
by hand and by token-hash; see the per-preset comments for the exact provenance
of each version.

PROVENANCE / STATUS
-------------------
These fixed lists are **superseded** by the automatic dataset-scaling mechanism
in ``train.py``: ``--n_per_rule_games N`` selects games from a universe
(``--n_per_rule_universe``, default the 3,474-game token-deduped pool) ranked or
stratified by rule count (``--n_per_rule_strategy``), with area and randomness
filters. New scaling work should use that flag rather than growing this file.

They are kept here (rather than deleted) because a number of run scripts under
``nca_wm/scripts/`` and recorded experiments still reference ``--games
scaling_gallery_vN`` by name, and the paper's reported runs were trained on
them. ``train.py`` merges :data:`SCALING_GALLERY_PRESETS` into its
``MULTI_GAME_PRESETS`` at import time, so the preset names resolve exactly as
before.

This module is intentionally dependency-free (pure data) so it can be imported
both by ``train.py`` and by tooling such as ``game_curriculum.py`` without
pulling in jax / the C++ backend.
"""

SCALING_GALLERY_PRESETS = {
    # scaling_gallery_v1: scaling_large (20) + ~20 extra gallery games filtered
    # for moderate complexity (n_rules ≤ 8, n_objects ≤ 12, max_level_area ≤ 20,
    # 1 ≤ n_levels ≤ 20). Sokoban_basic/Microban-style variants and lower-cased
    # duplicates are deduped, capitalized form preferred where both exist.
    "scaling_gallery_v1": [
        # scaling_large (20):
        "nekopuzzle", "notsnake", "blocks", "sokoban_basic", "sokoban_match3",
        "Zen_Puzzle_Garden", "Multi-word_Dictionary_Game", "kettle",
        "Travelling_salesman",
        "blank", "sumo", "the_undertaking", "wrappingrecipe",
        "Collapsable_Sokoban", "Love_and_Pieces", "actiontest",
        "rigidfail1", "scriptcross", "Modality", "constellationz",
        # +20 more gallery games (sorted by rules, then objects):
        "randomrobots",                # 1r,3o,1L
        "againexample",                # 1r,5o,1L
        "Microban",                    # 1r,5o,10L (capitalized)
        "naughtysprite",               # 2r,6o,1L
        "randomspawner",               # 2r,6o,1L
        "twolittlecrates1",            # 2r,6o,1L
        "rigid_11",                    # 3r,5o,1L
        "Long_Haul_Space_Flight",      # 3r,9o,13L
        "leftrightnpcs",               # 4r,5o,1L
        "twolittlecrates2",            # 5r,6o,1L
        "twolittlecrates3",            # 5r,6o,1L
        "twolittlecrates4",            # 5r,6o,1L
        "octat",                       # 5r,6o,8L
        "lunar_lockout",               # 5r,7o,4L
        "Stairways",                   # 6r,5o,3L
        "the_art_of_cloning",          # 6r,9o,1L
        "rigid_scott1",                # 7r,7o,1L
        "rigid_one_unlimited",         # 7r,8o,1L
        "Some_lines_were_meant_to_be_crossed",  # 7r,8o,7L
        "blockfaker",                  # 7r,11o,5L
    ],
    # scaling_gallery_v2: scaling_gallery_v1 (40) + 20 more gallery games at
    # higher rule/object counts. Filter: ≤20 rules, ≤20 objects, ≤30 max
    # level area, ≤30 levels. sokoban_basic_*-style pixel variants and
    # lowercase duplicates of existing canonical names omitted.
    "scaling_gallery_v2": [
        # scaling_gallery_v1 (40):
        "nekopuzzle", "notsnake", "blocks", "sokoban_basic", "sokoban_match3",
        "Zen_Puzzle_Garden", "Multi-word_Dictionary_Game", "kettle",
        "Travelling_salesman",
        "blank", "sumo", "the_undertaking", "wrappingrecipe",
        "Collapsable_Sokoban", "Love_and_Pieces", "actiontest",
        "rigidfail1", "scriptcross", "Modality", "constellationz",
        "randomrobots", "againexample", "Microban",
        "naughtysprite", "randomspawner", "twolittlecrates1",
        "rigid_11", "Long_Haul_Space_Flight", "leftrightnpcs",
        "twolittlecrates2", "twolittlecrates3", "twolittlecrates4",
        "octat", "lunar_lockout", "Stairways", "the_art_of_cloning",
        "rigid_scott1", "rigid_one_unlimited",
        "Some_lines_were_meant_to_be_crossed", "blockfaker",
        # +20 more games (sorted by complexity):
        "Pushing_It",                  # 7r,15o,1L
        "2D_Whale_World",              # 8r,7o,8L
        "MazezaM",                     # 8r,13o,30L
        "Ebony_&_Ivory",               # 9r,6o,1L
        "Singleton_Traffic",           # 10r,4o,6L
        "mazetest",                    # 10r,11o,1L
        "Slidings",                    # 10r,12o,11L
        "Lime_Rick",                   # 11r,11o,11L
        "riverpuzzle",                 # 11r,13o,1L
        "Midas",                       # 15r,13o,15L
        "rigid_parallel_many",         # 17r,10o,1L
        "Take_Heart_Lass",             # 17r,14o,12L
        "rigid_many_broken",           # 18r,11o,2L
        "Lightdown",                   # 18r,14o,8L
        "The_observer's_paradox",      # 18r,19o,6L
        "rigid_parallel_unlimited",    # 19r,9o,1L
        "MC_Escher's_Equestrian_Armageddon",  # 19r,14o,4L
        "Smother",                     # 19r,15o,16L
        "It_Dies_In_The_Light",        # 19r,16o,4L
        "Pushcat_Jr",                  # 19r,19o,8L
    ],
    # scaling_gallery_v3 (n=97): scaling_gallery_v2 deduplicated by token-hash
    # (drops blank ≡ sokoban_basic, Microban ≡ sokoban_basic, twolittlecrates4
    # ≡ twolittlecrates2 — three preset entries that the encoder cannot
    # distinguish under tokenize_game(encode_sprites=False)) plus 40
    # stratified additions from the 3,474-game dedup pool at
    # data/dedup_candidates_v2.json. Buckets: 12 low (1-3 rules ≤8 objs),
    # 16 mid (4-8 rules ≤12 objs), 8 high (9-15 rules ≤16 objs), 4 xhigh
    # (16-20 rules). Generated by nca_wm/scripts/propose_scaling_v3.py on
    # 2026-05-05.
    "scaling_gallery_v3": [
        # Deduped scaling_gallery_v2 (57):
        "nekopuzzle", "notsnake", "blocks", "sokoban_basic", "sokoban_match3",
        "Zen_Puzzle_Garden", "Multi-word_Dictionary_Game", "kettle",
        "Travelling_salesman", "sumo", "the_undertaking", "wrappingrecipe",
        "Collapsable_Sokoban", "Love_and_Pieces", "actiontest", "rigidfail1",
        "scriptcross", "Modality", "constellationz", "randomrobots",
        "againexample", "naughtysprite", "randomspawner", "twolittlecrates1",
        "rigid_11", "Long_Haul_Space_Flight", "leftrightnpcs",
        "twolittlecrates2", "twolittlecrates3", "octat", "lunar_lockout",
        "Stairways", "the_art_of_cloning", "rigid_scott1",
        "rigid_one_unlimited", "Some_lines_were_meant_to_be_crossed",
        "blockfaker", "Pushing_It", "2D_Whale_World", "MazezaM",
        "Ebony_&_Ivory", "Singleton_Traffic", "mazetest", "Slidings",
        "Lime_Rick", "riverpuzzle", "Midas", "rigid_parallel_many",
        "Take_Heart_Lass", "rigid_many_broken", "Lightdown",
        "The_observer's_paradox", "rigid_parallel_unlimited",
        "MC_Escher's_Equestrian_Armageddon", "Smother",
        "It_Dies_In_The_Light", "Pushcat_Jr",
        # Low-rule additions (12):
        "Time-reversed_Microban", "clone_cloned_clones!_by_sammytooch",
        "bell,_book,_and_candle_by_bagenzo", "zombie_bedtime_by_dan",
        "Dot_Puzzle_3.1", "cat_in_a_box_by_trevor",
        "no_name_yet_by_dustin_lane", "bit_treats_by_barefootengineer1",
        "Carnival_Shooter!", "sokobee_simples_by_matteomenapace",
        "shadow_by_unknown_author", "my_game_by_calvin_mcnamara",
        # Mid-rule additions (16):
        "neko_puzzle_by_increpare", "division_by_paul_gomez",
        "el_laberinto_dorado_by_skyemoure",
        "simple_block_pushing_game_by_maximyzer",
        "snowman_game_by_yuna", "the_adventurer_by_mose_e.eckman",
        "boxob_by_randomnamemn", "________________by_bagenzo",
        "pokemon_gooooo_by_drewdrewsan", "simple_foxes_by_torybash",
        "run_em_over_by_xxsethyxxxoultraomeg",
        "encuentra_a_tu_amigo_by_tobiasguzzo",
        "build_a_snowman_v2.0_by_arthur_yi",
        # `broken_by_beekie18` removed 2026-05-05 — its `[Crate ConveyUp] ->
        # again` rule causes the C++ A*/BFS search to hang indefinitely
        # (timeout_ms not honored once an unbounded `again` cycle starts).
        # See `feedback_broken_by_beekie18_hangs.md`. Investigate later.
        "constellation_by_jere_majava",
        "avaruus_tunkeutujat_by_janjulije",
        # High-rule additions (8):
        "Union_Move", "cyber_sokoban_by_eraykaan", "game_by_jobiden123",
        "nabokos_3_-_tech_demo_by_asynartesies", "buny_hoping_by_pickten",
        "drunkard_walk_by_unknown_author", "directional_insanity_by_zithral",
        "tron_by_lorapel",
        # Xhigh-rule additions (originally 4; reduced to 2 — the_master_zombie
        # and adventures_of_felix dropped 2026-05-05 because the games_metadata
        # `max_level_area` field stores only one dim, missing that
        # adventures_of_felix is 21×81 (1700 cells) and master_zombie is
        # 27×51 (1377 cells). Both blow JAX OOM during the first
        # training-step JIT compile (36 GiB allocation request). Mazetest at
        # 30×40 (1200 cells) is the largest known-working game in the v2
        # gallery; we keep below that ceiling now.
        "venganza_i_by_felipe_gomez", "spell_by_beekie18",
    ],
    # scaling_gallery_v4 (n=200): superset of scaling_gallery_v3, extending
    # the deduped pool further. Same stratified bucketing logic as v3 with
    # 143 stratified additions on top of the 57 deduped gallery_v2 games.
    # Generated by nca_wm/scripts/propose_scaling_v3.py --n_extra 143.
    "scaling_gallery_v4": [
        "nekopuzzle", "notsnake", "blocks", "sokoban_basic",
        "sokoban_match3", "Zen_Puzzle_Garden", "Multi-word_Dictionary_Game",
        "kettle", "Travelling_salesman", "sumo", "the_undertaking",
        "wrappingrecipe", "Collapsable_Sokoban", "Love_and_Pieces",
        "actiontest", "rigidfail1", "scriptcross", "Modality",
        "constellationz", "randomrobots", "againexample", "naughtysprite",
        "randomspawner", "twolittlecrates1", "rigid_11",
        "Long_Haul_Space_Flight", "leftrightnpcs", "twolittlecrates2",
        "twolittlecrates3", "octat", "lunar_lockout", "Stairways",
        "the_art_of_cloning", "rigid_scott1", "rigid_one_unlimited",
        "Some_lines_were_meant_to_be_crossed", "blockfaker", "Pushing_It",
        "2D_Whale_World", "MazezaM", "Ebony_&_Ivory", "Singleton_Traffic",
        "mazetest", "Slidings", "Lime_Rick", "riverpuzzle", "Midas",
        "rigid_parallel_many", "Take_Heart_Lass", "rigid_many_broken",
        "Lightdown", "The_observer's_paradox", "rigid_parallel_unlimited",
        "MC_Escher's_Equestrian_Armageddon", "Smother",
        "It_Dies_In_The_Light", "Pushcat_Jr",
        # +143 stratified additions:
        "bit_treat_by_barefootengineer1", "error_1_by_bvoq",
        "threat_level_midnight_by_vignesh", "the_magic_broom_by_radiosoap",
        "davib_by_thebigfive-381",
        "cat_adventure_deluxe!_demo_by_testethetestcat", "Opposition",
        "my_eyes!!!_by_explodedcoder", "a_game_by_password69",
        "bombero_en_acción_by_tevo66", "displacement_by_oliverwebsim",
        "bittle_bat_by_unknown_author", "match_threekoban_by_ezra_szanton",
        "my_game_by_amanda_howell", "puzzle_by_unknown_author",
        "super_mario_nazi_killer_by_koldo", "ditto_by_acercandrito",
        "the_far_away_danish_pastry_is_always_deliciouser_by_increpare_[variant_of_lexaloffles_neko_puzzle]",
        "prototype_1_by_caurso99", "herding_cats!!!1!_by_docchewkie",
        "robot_run_by_veeraalt", "sokobad_by_noacubestudio",
        "block_pushing_game_tutorial_2-4_by_unknown_author",
        "2to_by_yeah-and-no", "blockpush_by_tapcat",
        "simple_block_pushing_game_by_zaydiscool777",
        "neko_puzzle_by_increpare", "no_forbidden_symbols_by_increpare",
        "All_These_Damn_Crates", "bahçıvanın_çiçekleri_by_harunylc",
        "t-rex_hot_chocolate_by_therealeddawson",
        "zombie_blocks_by_aidan_quade", "the_poop_game_by_pontus_granstrom",
        "hanoi_by_sfiera", "feed_the_flames_by_callum_scott",
        "match_by_0creds", "Monster_block_push_game",
        "welcome_to_flatland_by_caitlin_weaver",
        "switch_switcher_by_increpare", "snow_day_by_forgdeer",
        "ambiguity_generalization_prototype_by_thebigfive-381",
        "avain_peli_by_my_name_here", "midnight_snack_by_lilianaaaaaaaa",
        "de+a+ch_by_razthepenguin",
        "variations_on_constellation_z_by_n_a",
        "rabbit_meets_camel_by_james_wood", "golem_five_by_henketime",
        "neko_puzzle_by_necrothaum",
        "puzzle_script_random_game_by_jeremy_labossiere",
        "custom_expulsion_by_chinbag", "swizzeroony_by_increpare",
        "lumberjack_run_by_nickromancer",
        "build_a_snowman_v1.0_by_kimchi2012",
        "impossible_pushing_-_level_2_by_stingby12",
        "push_pull_undo_redo_by_11pepi", "trample_by_jonathan_whiting",
        "simple_block_pushing_game_by_mirage-xel", "marathon_by_2sman",
        "simple_puzzle_by_paul", "maze_by_251119s",
        "build_a_snowman_v1.0_by_minjikim2013",
        "skyroad_by_sorendipitous", "interchange_by_pickten",
        "combo_by_nicholas_udell", "path_to_the_bee_by_jonbro",
        "Game_One", "One_Way_Street", "lasers_by_franklin_p._dyer",
        "on_thin_ice_by_gamez7_with_music_by_roccow",
        "robot_soccer_by_weiliuu", "Bridge-toggle_Maze",
        "flipping_by_mlipmanh", "bit_match!_by_unknown_author",
        "Lime_Richard", "naomis_challenge_4_game_winter_by_techythanh",
        "abocalipc_zumbi_br_by_robertjesus",
        "puzz-not_by_stew_hogarth",
        "simple_block_pushing_game_by_zacharybarbanell",
        "go_hunt_ψ™![beta]_by_puzzle00", "Festive_Lights",
        "Rolling-block_colour-zone_mazes",
        "build_a_snowball_v2_by_coconana9010",
        "remote_sokoban_by_poorlydrawncactus", "mr.hooper_by_yadel23",
        "blind_ninja_by_connorses", "sticky_v0.7_by_connorses",
        "green_by_laindey_denaige",
        "cathartic_colours_by_ronan_tumelty", "cow_rancher_by_bagenzo",
        "buny_hoping_by_pickten", "Solitaire",
        "mini_grapplenauts_by_jesse_purvis",
        "telekin_by_diego_cathalifaud", "Carnival_Shooter!",
        "minecraft_v1.0.1_by_coconana9010", "my_game_by_dio7225",
        "feed_the_cat_by_kim_cerna", "test_golfer_by_simonskvara",
        "maze_by_meena0303", "move_3_crates_next_to_eachother_by_alex_giger",
        "herding_cats!_by_albert24082408", "j_by_jack_gilbert",
        "schatkamer_by_jan-niestadt", "arregla_la_vereda_by_marcos-do",
        "test_by_finn_coffey",
        "simple_block_pushing_game_by_meddlingcactus",
        "cat_making_friends_by_wfykitty", "tall_player_by_henry-friedman",
        "my_game_is_better_than_your_game_by_nahcirn",
        "a_very_good_puzzle_by_nahcirn", "double_message_by_sevansevan",
        "weeeee_by_phuongstillhatework", "testi_by_janisamuli",
        "amor_en_tiempos_de_pandemia_by_i.enba",
        "space_invaders_by_e_is_cool___edited_by_fz0718",
        "two_coats_and_crates_by_henry-friedman",
        "ea_sports_fifa_2018_qatar_world_cup_edition!_by_john_q._futbol_and_ea_sports_of_course",
        "space_invaders_by_solafson",
        "platformer_thing_by_thesalamaderboy",
        "underpuzzle_by_pac2005",
        "eventually_this_is_gonna_be_zelda_by_soolseem",
        "endless_practice_by_plurmorant",
        "player_move,_pause,_badfish_move,_pause_by_jon-jon",
        "title_by_author", "rgb_by_michael_straeubig",
        "prototype_by_tebrown1011", "full_bloom_by_qwaffles",
        "beat_game_test_by_connorses_[loneship_games]", "Cell_Division",
        "test_by_katelyn_delta", "__by_jennybuckley",
        "drunkard_walk_by_unknown_author",
        "crossing_the_gap_by_barefootengineer1",
        "marbles_by_jcmiller11", "annoying_skull_by_qwaffles",
        "Colors", "puzzletracker_by_puzzlescriptfan34",
        "card_by_lintsungyueh",
        # `the_master_zombie_by_seth_archambault` dropped 2026-05-05: its
        # actual env shape is 27×51 (1377 cells) but games_metadata records
        # max_level_area=27 (only one dim), missing that it would OOM JAX.
        "ice_wolrd_by_scott_hughes",
        "test_teleportation_by_eoniz", "unlink_by_zacharybarbanell",
        "pathfinding_by_madball",
    ],
    # scaling_gallery_v5 (n=500): strict superset of scaling_gallery_v4
    # (199 games), extending it by 301 stratified additions drawn from the
    # 3,474-game dedup pool, disjoint from v4 by token-hash and disjoint
    # from Heldout-26. Bucket distribution of the additions: 90 low / 120
    # mid / 60 high / 30 xhigh (rule-count buckets matching the v3/v4
    # recipe). Generated by scripts/propose_scaling_v3 plus a manual
    # superset re-base; see data/scaling_gallery_v5_n500.json for the
    # full provenance record. The preset is built at module-import time
    # by concatenating scaling_gallery_v4 with the 301 new additions, so
    # the preset stays a strict superset of v4 even if v4 is later edited.
    "scaling_gallery_v5_extra": [
        "Minimalist", "mieti_by_jesse_ranta", "sokoban_horizontal",
        "box_mover_by_jessed926", "slider_by_holden", "ESCAPE!",
        "crate_navigation_by_thebigfive-381", "Time-reversed_Microban",
        "obeezyquest_by_obeezycraft", "wierd_pull_game_by_ben90y",
        "jonnedungeon_by_jarakko", "naborciM", "letters_to_santa_by_aki",
        "the_push_by_ric_cowley", "p_vs_z_by_jason_han",
        "eat_me_by_sarah_gp",
        "horizontal_block_pushing_game_by_stephen_lavelle",
        "sokkelo_by_ebjorm", "Escape_The_Haunted_Mansion", "______by_niki",
        "block_pushing_mini_game_by_aaron_huang",
        "block_pushing_game_by_stephanie",
        "penis_cloner_-_cem_special_edition_by_talha_kaya",
        "da_puzzle_game_by_joe", "manic_magellan_by_nonameghost",
        "cage_the_cat_by_alex", "portal_maze_-_equipo_8_by_cdinardi",
        "icebreaker_by_joaquin_cardozo", "lonely_sun_by_jack_guffey",
        "forest_by_arabella-jiang", "Labyrinth",
        "3_boxes_problem_by_archimedes", "hero_ball!!_by_kelly",
        "pugs_by_pug_lords", "Dot_Puzzle_3.1", "bunny_by_gianca33",
        "treat_maze_by_unknown_author", "mijn_puzzelspel",
        "my_game_by_atianyi1995", "herding_cats!_by_wushustyle",
        "Zombie_Invasion", "herding_cats!_by_n-matute95",
        "blocking_love_by_maura_kaiser", "dye_the_blocks_by_haptiikka",
        "ascendant_quest_by_thatwhichis", "puzzle_island_by_thelordvader55",
        "microban_by_pastimeshark",
        "player_with_2_sprites_and_restart_by_realshadowcaster",
        "forg_by_lanastepisnik", "jinja_no_neko_by_sharurin",
        "ice_by_system-fan", "ai-escape_by_cotidib",
        "into_the_inferno__working_by_united_states_of_asian",
        "meta_neko_demo_by_unknown_author",
        "find_the_cats!_by_iloveryujinandbingus", "catch_my_heart_by_mixi05",
        "ocd_creats_by_yadel23", "meteor_shooter_by_rodrigotorres28",
        "wilson,_cazador_de_vampiros_by_longfield4ever",
        "snow_target_by_puzzlevisst", "food_run_by_rafique_roberts",
        "the_infinite_maze_by_totally707",
        "el_laberinto_dorado_by_skyemoure",
        "you_have_to_light_the_floor_by_octoconnors", "true_spot_by_agucou",
        "fickle_companion_by_rywright93", "Teh_Interwebs", "mover_by_brodie",
        "kuzco_hechizado_by_sofia-gam", "Keys_and_Doors_0.1.0",
        "_simple__block_pushing_game_by_xligamer",
        "crazy_dog_walker_by_goldagent", "hi_by_therealmasterw",
        "let_me_survive_by_tianxiao_ren", "ludover_by_ariel",
        "puzzleball_by_josh_giesbrecht",
        "oskars_counter-step_maze_by_oskar_van_deventer__puzzlescript_by_mat_8e3f681d",
        "my_game_by_keyband", "my_game_by_gxner",
        "the_great_escape_by_florian_kastner", "bloks_by_spencer_mccauley",
        "snowman_game_by_arthur_yi", "posessor_by_dreamcastgh0st",
        "entierro_by_felipecabrera1804", "Stand",
        "snake_crate_by_jrigs-qu22",
        "Every_Three_Steps_You_Hit_a_Wall_Out_of_Nowhere",
        "pink_by_elitamasan", "captain_heauxbeaux_by_koser_dison",
        "sokoodd_by_monakrom", "Roots",
        "celebrity_[incomplete]_by_toph_wells",
        "policemen_and_zombies_by_dittoslash", "Ball_smashs_blocks",
        "random_robots_by_molsal", "neko_puzzle_by_brinaidk",
        "ayy_lmagnet_by_shembu9", "deeper_&_deeper_by_johnicholas",
        "oktoberfest_by_kubanameste", "pegs_by_zachary_sugano",
        "biuld_a_snowman_by_uwu", "hat_travelleraw_by_maria-shchurova",
        "prototype_1_by_janascheibner", "sheep_ship_company_by_lunardustx",
        "kong__island_defender_by_geojax", "example_puzzle_by_croubble",
        "build_a_snowman_1.0_by_puzzle00",
        "a_house_for_monsters_by_robin_gibson", "back_home_by_geralas",
        "cat_adventure_deluxe!_by_testethetestcat",
        "ranas_[v0.2]_by_argaane", "unicorn_on_acid_by_mira_lyster",
        "vampire_maze_by_tria1999", "yell_denier_by_nellie_dyer",
        "move_the_block_by_throwaway-github-visst", "Combine!",
        "Black_&_White", "crate_++_test_by_tjc_games",
        "crate_mover_by_jessed926", "build_a_snowman_v1.0_by_therealmasterw",
        "build_a_snowman_v2.0_by_ylimes10",
        "sticky_cube_[leaves_pack]_by_staffhook",
        "highschool_survivors_by_gianca33", "wild_fire_by_ad-18-code",
        "blockill_puzzlescript_version__by_isaac_d.idea_by_ivan",
        "mirror_blocks_by_edalcmagal", "laser_pointer_by_n_a",
        "microban_by_nickneim", "catch_my_heart_by_khavlz",
        "rigid_player_by_not-there-yet", "________________by_s0ra",
        "moleg_prototype_by_zacharybarbanell",
        "monsterescape_by_lordkeker31",
        "peewees_garden_by_gabriele_maddaloni", "Game_101",
        "dumb_box_thing_by_mokesmoe", "Switcheroo",
        "billy_bobs_variable_vertigo_by_monakrom",
        "stutterstep_70s_style_by_andreas", "sport_by_ninabirb",
        "lesson_in_trickery_by_ncrecc", "mine_mania!_by_jarred_hanson",
        "my__game__by_elisa,_fran,_martin,_muhsia",
        "super_push_luigi__green_guy__bros_by_playergame123",
        "my_by_jacobrogers2", "hack_it!_by_your_mother",
        "boxman_v0.7.2_by_rando2048", "manners_gold_by_gancuililuo",
        "turbanator_2_-_the_legend_of_ronik_and__a6936f1f_by_davidchernenko",
        "cool_stuff_by_wunwun3", "cover_the_hole_by_rmvelez",
        "Tractor_Beam_Sokoban9",
        "atrapa_pajaros_20_000_by_maaaaaaaaaximilianoj",
        "block_fakers_puzzle_garden_by_stingby12", "whack-a-deer_by_learus",
        "cloaky_guy_by_your-die", "Line_of_Sight", "santoban_by_eckelito",
        "game_by_jobiden123", "beast_box,_issue_2_by_sus1d1p",
        "________________by_jackkutilek",
        "build_a_snowman_v1.4_by_coconana9010",
        "snowman_building__real_phisics__final_product_by_gavinleong08",
        "birthday_candles_by_franklin_p._dyer", "Yin-Yang",
        "maze_editor_by_realshadowcaster", "sticky_you_by_ojamajoshirami",
        "build_a_snowman_by_williamzhong157", "golf_v8.41_by_editmouse",
        "my_elephant_by_klianc09", "kastid+_by_tseik6", "2048_by_ramified",
        "sock_oddity_abacus_by_zithral", "puzzle_maze_v0.1b_by_e_is_cool",
        "pompoenen_rollen_by_jan-niestadt", "shoving",
        "wittest_by_razthepenguin", "bounce_&_eat_by_daniel_yaskin",
        "shooting_sim_-_upload_by_urlocalrussell", "Mitosys",
        "lab2adventuregame_by_lindsaymacdowell", "Goblin_Hooblob",
        "smother_by_steven_darragh_rach_ricky",
        "glitchstrike__switchback_by_soulcakelive",
        "dungeon_roll_by_janwatever", "this_is_the_only_level_by_e_is_cool",
        "lime_rick_with_jumps_7_high_instead_of_3_by_tommi_tuovinen_edited_by_diribigal",
        "hivemind_by_itsmichal", "elements_by_bob_ross",
        "make_a_snowman_by_matthewsuncodinghomework",
        "fernwirkung_by_jakwolf", "corpse_sokoban_by_poorlydrawncactus",
        "Match_Flow", "silver_lungs", "horror_at_diaspi_by_oori",
        "32_[demo_3]_by_sefcearcatpro", "Puzzleboi",
        "drop-o-trons!_by_alxama", "carrera_de_galgos_by_yogui1972",
        "cidade_by_unknown_author", "maze_generator_by_matthieuhaller",
        "sumo_by_marek_jarzabek", "despachante_de_aduana_by_nachlord996",
        "midas_by_riotkings", "simple_gravity_game_by_unexian",
        "vence_en_el_espacio_by_rmayans", "a-maze-ing_game_by_nkira4869",
        "gdd301_canera_follow_game_in_class_by_gamesatqu",
        "my_game_by_brunogarcilazo", "test_game_1_by_jackbarinaga",
        "lil_game_by_alina_st_v", "the_legend_of_the_dungeon_by_tiex",
        "riblix_by_emoti-w-1337", "remembrance_by_philippa_warr",
        "13_skulls_by_unknown_author", "drawing_by_pap4qlxxlevb",
        "infiltrating_area_51_by_danlittle211",
        "simple_block_pushing_game_by_bananamath",
        "basic_programming_tutorial_-_incomplete_by_stuarts_pixel_games",
        "little_dungeons_by_darkhog", "my_game_by_calvin_mcnamara",
        "sapo_game_by_julilopezzz",
        "multi-word_dictionary_game_by_increpare",
        "random_robots_by_lee2sman", "double_movin_by_jcmiller11",
        "chucker_by_connorses",
        "leeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee_by_5381_and_pito",
        "squares_by_tim_inman", "box_game_template_by_franklin_p._dyer",
        "my_game_by_heidevacht", "simple_block_pushing_game_by_nacmacfigl",
        "Deadly_Cat_Maze", "spawning_madness_by_cagatayaydin",
        "havrest_trees_by_laurieakihiko",
        "sokoban_adder_-_no_snakes!_by_beekie18",
        "simple_block_pushing_game_by_wavak-creations",
        "crate_dashing_by_pazzaz", "my_first_puzzle_by_spinnaker2",
        "skwiffle_by_strickinato", "ice_slide_demo_by_michael_fairley",
        "vi-seth_starting_game_by_vi-seth_bak",
        "big_boy_by_franklin_p._dyer", "slasher_by_mario-egocheaga",
        "simple_block_swapping_game_by_ncrecc",
        "cupcakes_of_the_dead_by_krpopp", "triptych_by_beekie18",
        "translated_parts_thing_by_sus1d1p", "eye_eye_eye_by_increpare",
        "catch_das_kapital!_by_jere_majava",
        "violent_puzzle_game_by_notajumbleofnumbers", "CC2",
        "procedural_generation_dungeon_by_tha_cuber",
        "sliding_by_coreyhardt", "moving_thing_by_thesalamaderboy",
        "minigame_prototype_by_rcmichaels",
        "more_fixed_maze_gen_by_unknown_author",
        "simple_block_pushing_game_by_heyimlate",
        "excute_prisoners_by_mean_person", "Turing_Machine",
        "untitled_by_11pepi", "cannonfall_by_equalzdee",
        "slayer..or_something_by_carter_gleason",
        "diagonal_sokoban_by_thebigfive-381",
        "no_forbidden_symbols_2_by_increpare", "drill_by_chinbag",
        "turing_machine_by_ashlikatt", "virus_by_wunwun3",
        "diagonal_test_by_connorses", "roguekoban_by_vivribbon", "Shoop",
        "hurder_by_riley_van_etten", "scarf_of_shootign_star_by_s0ra",
        "prototipo_de_movimiento_de_enemigos_by_constanza_dibueno",
        "ice_example_by_jcmiller11", "vacuum_thing_by_jcmiller11",
        "streamlined_plastic_cage_by_zacharybarbanell",
        "simple_block_pushing_game_by_xdxdxdlol",
        "directional_insanity_by_zithral",
        "i_made_the_game_where_you_kill_a_goblin_5b028ece_by_connorses",
        "lexoban_007_by_rubzo", "platforming_engine_test_by_gamez7",
        "microban_2.0_by_baba-is-text", "platformer_test_by_bananamath",
        "rotating_cop_by_playfulsystems",
        # `up_and_down_forces_by_rosden` removed 2026-05-08: contains
        # `[ ] -> again` (always-fire, always-restart) which makes the
        # C++ A* / BFS collector loop indefinitely; same pathology as
        # `broken_by_beekie18` (see feedback_broken_by_beekie18_hangs).
        # Hung Train-500 cond data collection at game 481/497 for 4.5h
        # before the timeout fired. Investigate later.
        "thought_experiment_by_shylicks",
        "simple_block_pushing_game_by_kramff", "connections_by_giles",
        "the_master_zombie_by_seth_archambault", "tbd_by_jackkutilek",
        "block_faker_by_mikechecker", "Ice_Breaker",
        "circular_shades_by_zithral", "slideing_movement_test_by_kalixtan",
        "simple_block_pushing_game_by_variousauthors",
        "healthcare_simulator_by_micaherb", "match-explode_by_stingby12",
        "origami_game_by_s0ra",
        "dave_phillips_color_path_#3___puzzlesc_358f80cf_by_eduardo_alonso______··",
        "magicdragon_[ps_depth_test]_by_ladyleia",
        "1-2-3_mazes_by_erich-friedman"
    ],
}


# Build scaling_gallery_v5 (Train-500) as a strict superset of v4 at module
# import time so the relationship to v4 stays correct if v4 is later edited.
SCALING_GALLERY_PRESETS["scaling_gallery_v5"] = (
    list(SCALING_GALLERY_PRESETS["scaling_gallery_v4"])
    + list(SCALING_GALLERY_PRESETS["scaling_gallery_v5_extra"])
)
