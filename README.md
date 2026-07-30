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
- Maze memory with closest-frontier exploration and reset support.
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
`east`, `south`, and `west`. On the Maze tab, choose **Set walkable ground**
and capture one large floor area while excluding UI and animated scenery.

The transition watcher downsamples that area into a 12 by 8 grid and samples it
independently of the normal decision interval. Walking normally produces a
connected stream of changed chunks. A transition candidate is recorded when
three or more spatially separated chunks change, when those changes span at
least four grid cells, or when at least 25% of usable chunks change together.
The room move is committed after the grid stabilizes for two samples. Chunks
that change repeatedly while no move is pending are learned as volatile and
ignored. Combat cancels the pending move because the battle occurs on the
current tile.

The grid size, chunk difference, separation, coverage, stability, sampling,
volatility, and 12-second recovery timeout are configurable with the
`walkable_*`, `transition_sample_seconds`, and
`transition_pending_timeout_seconds` settings. Pending movement suppresses
additional exploration clicks only; it does not block actions in other states.

The **Sprites** tab manages the boss template. Legacy player-pose captures are
listed as unused and can be previewed or deleted; player tracking is no longer
part of scene-transition detection.

On the **Maze** tab, label the visible exits with the N/E/S/W checks and choose
**Submit checked tile layout**. The bot stores a SHA-256 identity made from the
raw, full-resolution color pixels in all four `*_passage` regions, plus the
checked-exit set, in `tile_layouts.json`. This dictionary persists across
randomized mazes and bot restarts.

Existing configurations that already contain all four directional
`scene_change_*` captures continue including those pixels in tile fingerprints.
Those regions are no longer used to detect transitions, but retaining them
keeps previously submitted exact layout IDs and comparison descriptors usable.
New configurations use the four passage regions alone.

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

The solver has no fixed coordinate goal. It takes an untried exit in the
current room, then uses the shortest known route to the nearest mapped room
that still has an unexplored branch. This closest-frontier strategy exposes new
branches quickly. A detected boss overrides the frontier recommendation.

### Boss detection

Add an exploration visual region named `boss_search_area` covering every place
where the boss sprite can appear. In the Maze tab, choose **Capture boss
sprite** and drag tightly around the boss itself without including room
background.

During exploration, the bot always selects the click zone named by the
closest-frontier maze recommendation. A boss match overrides that recommendation
and always selects the `mid` click zone. The learned action policy does not
override either exploration behavior. The Maze tab labels the boss action as
`BOSS`.

### Training the action policy

The policy treats each clickable zone in a state as one possible action. It
summarizes the color and brightness of the state's named visual regions and
uses that context to predict a zone.

1. Select **Training** on the Control tab.
2. On the Training tab, enable **Learn from my clicks while in Training mode**.
3. Play normally. The green marker shows the policy prediction.
4. Click one of the configured zones. A prediction match earns reward 1; a
   different configured-zone click earns reward 0 and teaches the selected
   action.

Clicks outside configured zones are ignored by training. The learned model and
statistics are stored in `policy.json`. Retaking or adding visual regions or
click zones changes the model shape for that state, so its learned weights
restart automatically. Keep Live mode off until the predictions are reliable.

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
