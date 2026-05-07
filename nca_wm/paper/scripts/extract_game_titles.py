"""Extract human-readable titles from PuzzleScript .txt files for the appendix
preset lists. Search order: custom_games -> gallery_games ->
data/scraped_games -> data/scraped_games_increpare. Outputs JSON map.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SEARCH_DIRS = [
    REPO_ROOT / "custom_games",
    REPO_ROOT / "gallery_games",
    REPO_ROOT / "data" / "scraped_games",
    REPO_ROOT / "data" / "scraped_games_increpare",
]

# Lists copied verbatim from nca_wm/train.py and data/heldout_v4_n30.json so
# this script is self-contained.

TRAIN_14 = [
    "nekopuzzle", "notsnake", "blocks", "sokoban_basic", "sokoban_match3",
    "Zen_Puzzle_Garden", "Multi-word_Dictionary_Game", "kettle",
    "Travelling_salesman",
    "Collapsable_Sokoban", "Love_and_Pieces", "actiontest",
    "scriptcross", "Modality",
]

TRAIN_94_BASE_57 = [
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
]

TRAIN_94_LOW = [
    "Time-reversed_Microban", "clone_cloned_clones!_by_sammytooch",
    "bell,_book,_and_candle_by_bagenzo", "zombie_bedtime_by_dan",
    "Dot_Puzzle_3.1", "cat_in_a_box_by_trevor",
    "no_name_yet_by_dustin_lane", "bit_treats_by_barefootengineer1",
    "Carnival_Shooter!", "sokobee_simples_by_matteomenapace",
    "shadow_by_unknown_author", "my_game_by_calvin_mcnamara",
]

TRAIN_94_MID = [
    "neko_puzzle_by_increpare", "division_by_paul_gomez",
    "el_laberinto_dorado_by_skyemoure",
    "simple_block_pushing_game_by_maximyzer",
    "snowman_game_by_yuna", "the_adventurer_by_mose_e.eckman",
    "boxob_by_randomnamemn", "________________by_bagenzo",
    "pokemon_gooooo_by_drewdrewsan", "simple_foxes_by_torybash",
    "run_em_over_by_xxsethyxxxoultraomeg",
    "encuentra_a_tu_amigo_by_tobiasguzzo",
    "build_a_snowman_v2.0_by_arthur_yi",
    "constellation_by_jere_majava",
    "avaruus_tunkeutujat_by_janjulije",
]

TRAIN_94_HIGH = [
    "Union_Move", "cyber_sokoban_by_eraykaan", "game_by_jobiden123",
    "nabokos_3_-_tech_demo_by_asynartesies", "buny_hoping_by_pickten",
    "drunkard_walk_by_unknown_author", "directional_insanity_by_zithral",
    "tron_by_lorapel",
]

TRAIN_94_XHIGH = [
    "venganza_i_by_felipe_gomez", "spell_by_beekie18",
]

TRAIN_199_EXTRA_142 = [
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
    "ice_wolrd_by_scott_hughes",
    "test_teleportation_by_eoniz", "unlink_by_zacharybarbanell",
    "pathfinding_by_madball",
]

HELDOUT_30 = [
    "in_the_way_by_giovanni_mota", "cyberpunk_2020_by_gzhao-jpg",
    "my_nana_ate_my_kid_brother_by_ruth_williams",
    "ice_sliding_by_by_ian_cox", "block_game_by_emily_dresden",
    "kitten_rescue_by_unknownprodigy-beep",
    "kyles_chesse_cuisine_by_techythanh",
    "the_best_2d_game_ever_created_by_cwm_gaming",
    "monophobic_multiban_by_increpare", "charge_the_bot_by_tiddly5",
    "build_a_snowman_challenges_by_flyingfroakie",
    "the_single_flame_by_hassan_masood", "pacman__by_cora",
    "escape_by_shawn_martin", "megabot_invades!_by_louis_g",
    "link_by_jeffjeff123456", "________________by_ncrecc",
    "watchers_ritual_by_purpledragon17",
    "gdd301_demo_game_by_emma_hubbell", "Surround_Yourself_With_Dogs",
    "haberdashery_by_lee2sman", "ghosts_by_henry-friedman",
    "a_snowballs_chance_in_hell_by_iznaut", "hedgehogger_by_increpare",
    "cat_adventure_2_by_whenyoucando", "Hamiltwo",
    "headless_people_problems_by_monakrom",
    "break_out_of_the_mine_by_jja_i.e._juan,_jose_&_andre",
    "Heroes_of_Sokoban_-_Ancient_Japan", "angize_by_ali_nikkhah",
]


def find_file(name: str) -> Path | None:
    for d in SEARCH_DIRS:
        p = d / f"{name}.txt"
        if p.exists():
            return p
    return None


_TITLE_RE = re.compile(r"^\s*title\s+(.+?)\s*$", re.IGNORECASE)
_AUTHOR_RE = re.compile(r"^\s*author\s+(.+?)\s*$", re.IGNORECASE)


def extract_title_author(p: Path) -> tuple[str | None, str | None]:
    title = author = None
    try:
        with p.open(encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i > 50:
                    break
                if title is None:
                    m = _TITLE_RE.match(line)
                    if m:
                        title = m.group(1)
                if author is None:
                    m = _AUTHOR_RE.match(line)
                    if m:
                        author = m.group(1)
                if title and author:
                    break
    except OSError:
        pass
    return title, author


def fallback_title(name: str) -> str:
    """Used when a game's source .txt is not on disk: convert filename
    to a human-readable approximation by replacing underscores with spaces
    and stripping the trailing _by_<author> suffix."""
    body = name
    for sep in ("_by_", " by "):
        if sep in body:
            body = body.split(sep, 1)[0]
            break
    return body.replace("_", " ").strip()


def latex_escape(s: str) -> str:
    """Minimal LaTeX-escape: characters that can occur in PuzzleScript
    titles and need escaping outside of math mode. Game titles
    contain &, _, #, %, $, but rarely backslashes or curly braces."""
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "_": r"\_",
        "#": r"\#",
        "%": r"\%",
        "$": r"\$",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "Ψ": r"$\Psi$",
        "ψ": r"$\psi$",
        "™": r"\texttrademark{}",
    }
    out = []
    for ch in s:
        out.append(repl.get(ch, ch))
    return "".join(out)


def display_title(entry: dict, t_collisions: set, ta_collisions: set) -> str:
    """Render a single entry. Bare 'Title' if unique; 'Title (Author)'
    if title alone collides; 'Title (Author; filename)' if (title, author)
    collides too (boilerplate PuzzleScript demo titles)."""
    title = entry["title"] or "?"
    author = entry["author"]
    name = entry["name"]
    t_key = title.lower()
    ta_key = (title.lower(), (author or "").lower())
    # Several authors used a row of underscores as a placeholder title;
    # in italics the bare underscores merge into a baseline rule, so wrap
    # the literal title in quotes to delimit it. Do this *after* computing
    # collision keys so the quoted form doesn't bypass dedup.
    if title and set(title) <= {"_"}:
        title = f"``{title}''"
    if ta_key in ta_collisions:
        return f"{title} ({author}; {name})" if author else f"{title} ({name})"
    if t_key in t_collisions and author:
        return f"{title} ({author})"
    return title


def emit_latex_list(entries: list, t_collisions: set, ta_collisions: set) -> str:
    rendered = [
        latex_escape(display_title(e, t_collisions, ta_collisions))
        for e in entries
    ]
    rendered = [f"\\textit{{{r}}}" for r in rendered]
    return ", ".join(rendered)


def main() -> None:
    presets = {
        "Train-14": TRAIN_14,
        "Train-94 base 57": TRAIN_94_BASE_57,
        "Train-94 low": TRAIN_94_LOW,
        "Train-94 mid": TRAIN_94_MID,
        "Train-94 high": TRAIN_94_HIGH,
        "Train-94 xhigh": TRAIN_94_XHIGH,
        "Train-199 extra 142": TRAIN_199_EXTRA_142,
        "Heldout-30": HELDOUT_30,
    }
    out: dict[str, list[dict]] = {}
    missing = []
    for preset, names in presets.items():
        out[preset] = []
        for name in names:
            p = find_file(name)
            if p is None:
                missing.append(name)
                title = fallback_title(name)
                author = None
                source = None
            else:
                title, author = extract_title_author(p)
                source = str(p.relative_to(REPO_ROOT))
                if title is None:
                    title = fallback_title(name)
            out[preset].append({
                "name": name,
                "title": title,
                "author": author,
                "source": source,
            })

    # Collision detection: count distinct *filenames* sharing the same
    # title (or title+author). Train-14 is a subset of Train-94, so we
    # dedup by filename first, otherwise every game looks like it
    # collides with itself.
    seen: dict[str, dict] = {}
    for entries in out.values():
        for e in entries:
            seen.setdefault(e["name"], e)
    t_counts: dict[str, int] = {}
    ta_counts: dict[tuple, int] = {}
    for e in seen.values():
        t = (e["title"] or "").lower()
        a = (e["author"] or "").lower()
        t_counts[t] = t_counts.get(t, 0) + 1
        ta_counts[(t, a)] = ta_counts.get((t, a), 0) + 1
    t_collisions = {t for t, c in t_counts.items() if c > 1}
    ta_collisions = {ta for ta, c in ta_counts.items() if c > 1}

    mode = sys.argv[1] if len(sys.argv) > 1 else "json"
    if mode == "json":
        json.dump(out, sys.stdout, ensure_ascii=False, indent=2)
    elif mode == "latex":
        for preset, entries in out.items():
            print(f"% === {preset} ({len(entries)} games) ===")
            print(emit_latex_list(entries, t_collisions, ta_collisions))
            print()
    else:
        print(f"unknown mode: {mode}", file=sys.stderr)
        sys.exit(2)

    if missing:
        print(f"\n# WARNING: {len(missing)} games missing on disk:",
              file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)


if __name__ == "__main__":
    main()
