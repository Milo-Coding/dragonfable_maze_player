# DragonFable Bot

A safety-first Windows desktop bot foundation based on `Idea.md`.

## What works now

- Always-on-top control panel with **Idle**, **Training**, and **Live** modes.
- Global configurable mode shortcuts. Thumb mouse button 1 enables Training
  and thumb mouse button 2 enables Live by default; reassign either from the
  **Controls** tab.
- Global **Escape** emergency stop.
- Primary-monitor capture and template-based state detection.
- A second mode check immediately before every click.
- State-specific rectangular click allowlists.
- Maze memory with bounded-depth branch exploration and reset support.
- Lobby start action hook.

Live mode cannot click until both a target and a click zone are configured. The
current decision engine intentionally only implements the lobby hook; combat and
exploration require screenshots and calibration from the actual game.

## Install and run

Use Python 3.11 or 3.12 on Windows:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item config.example.json config.json
python run_bot.py
```

Press **Escape** at any time to return to Idle. Test Training mode thoroughly
before enabling Live.

## Configuration

Coordinates are absolute primary-screen pixels.

The easiest setup is through the control panel:

1. Put the game on the screen you want to capture.
2. Choose **Capture state anchor**, enter the state name, then drag around a
   distinctive visual feature.
3. Choose **Add allowed click zone**, enter the state, then drag around the
   area where the bot may click.
4. For the lobby, choose **Set lobby start point** and click the center of the
   quest button.

The tool freezes the current screen while you select. Captured state templates
are written to `templates/`, and all coordinates are saved to `config.json`.

### Exploring and combat visual regions

Open the **Calibration** tab and use **Add visual region**. Give each region a
stable descriptive name. Useful exploration regions include:

- `scene_change_north`, `scene_change_south`
- `scene_change_east`, `scene_change_west`
- `north_passage`, `east_passage`, `south_passage`, `west_passage`
- `room_type`
- Any indicator that changes when an exit is blocked or already used

Useful combat regions include:

- `player_health`, `player_mana`
- `enemy_1_health`, `enemy_2_health`
- `enemy_count`
- One availability/cooldown region for each action

The table shows all anchors, clickable zones, visual regions, and the lobby
target. Select an item to preview, retake, rename, or delete it. Custom states
can also be added or removed from this tab.

For maze tracking, rename the four exploration click zones exactly `north`,
`east`, `south`, and `west`. Use the Maze tab's four **Set** buttons to capture
the corresponding room-entry edges. Avoid animated scenery where possible.

Clicking an exit starts a pending movement window and blocks further exploration
clicks. If combat begins, the pending move is canceled and the maze position
does not change. Otherwise, the saved player sprites are used to confirm the
character has appeared and remained still for several samples inside the
expected destination entryway (the edge opposite the clicked direction). Prior
movement does not have to be observed. The bot then advances in the clicked
direction and adds the destination tile. This works for similar or
pixel-identical rooms because room appearance is not the transition evidence.

Movement distance, stopped sample count, sample rate, and timeout are controlled
by `transition_player_movement_pixels`, `transition_player_stable_frames`,
`transition_sample_seconds`, and `transition_pending_timeout_seconds`.
Pending movement suppresses additional exploration clicks only; it does not
block actions in other states.

The **Sprites** tab manages the boss template and player appearance templates.
Capture enough player poses/outfits for the character to remain detectable
while walking and while standing at each entryway.

On the **Maze** tab, label the visible exits with the N/E/S/W checks and choose
**Submit checked tile layout**. The bot stores a SHA-256 identity made from the
raw, full-resolution color pixels in all four `*_passage` regions, plus the
checked-exit set, in `tile_layouts.json`. This dictionary persists across
randomized mazes and bot restarts.

Existing directional `scene_change_*` captures remain part of tile fingerprints
so previously submitted exact layout IDs and comparison descriptors stay
compatible. They now also provide the four entry-edge images used by transition
detection.

Each dictionary entry has a stable ID such as `layout_003`. Encountered maze
tiles store a reference to that ID, shown as `L003` on the grid. An unseen
exact hash is provisionally linked to the submitted layout with the lowest
compact-image difference and is shown as `~L003`. This gives the solver a
temporary layout immediately; an exact match or a true layout submission
replaces the provisional link. Updating a
submitted layout's checked exits updates every currently mapped tile linked to
that layout. Re-saving a tile whose visuals match a different layout replaces
only that tile's link.

A submission updates an existing record only when all four passage regions produce
the exact same pixel hash. Any pixel difference, however small, creates a new
layout ID. Nearest-image matching never creates or overwrites an ID; it is only
a provisional fallback for an unseen tile. Legacy reduced-feature entries are
preserved but excluded from exact and nearest matching until they are submitted
again with current comparison data.

Use **Save checked exits to current maze tile** separately when you want the
checked exits to authoritatively replace the current tile's inferred paths.
Matching a submitted layout automatically adds its saved paths to unconfirmed
tiles. The 10 by 10 grid starts at `(0,0)`, shows the current tile in green, and
the solver's recommendation in yellow.

The solver has no fixed coordinate goal. At a branch, it estimates each exit's
maximum possible depth by counting the unknown 10 by 10 grid cells reachable
without crossing a mapped room, and tries the smallest region first. This
prioritizes short, constrained branches where the dead-end boss room is more
quickly found or ruled out. When the current room is exhausted, it uses the
shortest known route to another mapped room with an unexplored branch. A
detected boss overrides the maze recommendation.

### Teleporter routing

The solver uses one teleporter as a branch checkpoint. At a tile with multiple
unexplored exits, it places the teleporter and explores the shorter-depth path
first. It keeps that anchor until its branch tile is exhausted instead of
replacing it at a nested branch. When the current path is exhausted, it returns
directly to the anchored tile while that tile still has another unexplored exit.
It does not use the teleporter as a general shortcut to unrelated frontiers,
which avoids repeated back-and-forth teleports. The placement and return clicks
are execution steps only and have zero pathfinding cost.

Use the **Teleporter** tab to capture the three placement zones and two return
zones in click order. All steps must be configured before the corresponding
action is enabled. Teleporter sequences use the shorter
`teleporter_click_interval_seconds` delay (0.25 seconds by default), while
ordinary live actions keep their normal delay. The maze display marks the
current anchor with `T`.

### Boss detection

Add an exploration visual region named `boss_search_area` covering every place
where the boss sprite can appear. In the Maze tab, choose **Capture boss
sprite** and drag tightly around the boss itself without including room
background.

During exploration, the bot always selects the click zone named by the maze
recommendation. A boss match overrides that recommendation and always selects
the `mid` click zone. The learned action policy is not evaluated or trained
during exploration. The Maze tab labels the boss action as `BOSS`.

Boss detection also interrupts an in-progress teleporter placement or return.
The discovered tile is retained in maze memory and marked `B` on the grid. If
the player is later displaced, routing back to that known boss tile takes
priority over exploration. Boss discovery permanently disables teleporter
placement and return for the remainder of that maze.

### Mog detection

Capture the Mog sprite and its menu close zone on the **Sprites** tab, then set
the exploration `mog_search_area`. During exploration, a Mog match overrides
movement and clicks `mid` once. Because the exploration anchor remains visible
in the Mog menu, the bot explicitly clicks the calibrated Mog close zone on its
next action instead of relying on scene classification. The tile is remembered
for the current maze so the same Mog is not repeatedly triggered. Boss
detection retains priority over Mog detection.

### Training the action policy

The policy is used only for combat, where each clickable zone is one possible
action. Exploration is controlled entirely by the maze solver and boss
detection. Any other scene with exactly one configured click zone selects that
zone directly without prediction or training. The combat policy
summarizes the color and brightness of the state's named visual regions and
uses that context to predict a zone.

1. Select **Training** on the Control tab.
2. On the Training tab, enable **Learn from my clicks while in Training mode**.
3. Enter combat and play normally. The green marker shows the combat policy
   prediction.
4. Click one of the configured zones. A prediction match earns reward 1; a
   different configured-zone click earns reward 0 and teaches the selected
   action.

Non-combat clicks and clicks outside configured combat zones are ignored by
policy training. The learned model and statistics are stored in `policy.json`.
Retaking or adding combat visual regions or click zones changes the combat
model shape, so its learned weights restart automatically. Keep Live mode off
until the predictions are reliable.

Visual regions and clickable zones can be temporarily disabled from the
**Calibration** tab with **Enable / Disable**. Disabled items retain their
names and coordinates in `config.json`, remain listed with `Enabled: No`, and
can be restored later. Disabled visual regions are excluded from policy
features and visual detectors. Disabled click zones are excluded from policy
actions and the live-click allowlist. Changing either enabled set changes the
combat model shape in the same way as adding or removing a region.

- `regions`: named screen rectangles used for visual input. State anchors are
  named `menu_anchor`, `lobby_anchor`, `exploring_anchor`, and `combat_anchor`.
- `templates`: maps a state name to a cropped reference PNG.
- `click_zones`: maps each state to rectangles where clicks are permitted.
- `lobby_start_point`: the center of the Start Quest button.

Example:

```json
{
  "regions": {
    "lobby_anchor": {"left": 700, "top": 180, "width": 200, "height": 100}
  },
  "click_zones": {
    "lobby": [{"left": 800, "top": 650, "width": 180, "height": 60}]
  },
  "templates": {
    "lobby": "templates/lobby.png"
  },
  "lobby_start_point": {"x": 890, "y": 680}
}
```

## Recommended build order

1. Capture and label examples of the four game states.
2. Calibrate state anchors and click zones.
3. Add exploration exit regions and movement click points.
4. Add combat action availability plus health/mana readers.
5. Record Training-mode decisions and outcomes.
6. Only then enable Live for one state at a time.

For variable game screens, begin with image templates and color/fill
measurements. Once a labeled screenshot set exists, state classification and
combat scoring can be upgraded to learned models without changing the safety
controller.
