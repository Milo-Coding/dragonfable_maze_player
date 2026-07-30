I would like to design a bot capable of playing a simple video game about exploring a maze and fighting. It will need to see and click in my primary monitor to do this.



The bot must be able to click at certain points on the screen to interact with the game

The bot must be restricted from clicking if i press the escape key



the bot should have three modes, idle, training, and live

in idle, the bot will not run but the control menue will still appear

in training, the bot will display with a green circle where it would like to click

in live, the bot will actually click where it would like to



The bot should be able to detect what state the game is in from among menu, lobby, exploring, and combat

In the lobby, the bot will need to click one button to start the quest. it should also clear any knowledge about the maze as the maze is randomized each time the quest starts. The lobby will always be identicle with no variance

in combat, the bot should consider which actions are avalable to it and select the best one. it will need to be able to see if an action is avalable, see its own health and mana, and see the enemy/enemies health and mana (and how many enemies there are). There are three possible types of enemies and they sometimes attack two on one.

while exploring, the maze will be restricted to a 10 x 10 grid with each maze tile having the possability of a north, east, south, and west passage. the bot will need to be able to identify when a passage is avalable as well as if it has explored that passage yet. The bot should explore new tiles until it encounters the boss of the quest. to this end, the bot must maintain a record of explored tiles in the maze (these will be reset when it returns to the lobby). The bot should be able to identify the type of tile it is in as well as which exit it needs to take to best explore the maze.

there are several menus but nearly all of them have only one button that needs to be clicked. the bot must click these buttons



Within one of these four states, I want to be able to designate which areas of my screen provide relevant data. i also want to designate where the bot is allowed to click in each screen.

there are many possible configurations for the exploration and combat states so an adaptable or learned apporach would be best



the bot should be able to capture images directly and let me designate important areas with my mouse